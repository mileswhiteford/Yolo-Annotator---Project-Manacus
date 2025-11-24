import sys
from PyQt5.QtWidgets import (
    QGraphicsPixmapItem,
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QPushButton,
    QGraphicsView,
    QGraphicsScene,
    QLabel,
)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QRectF 
import cv2
import argparse
import os
from .bounding_box import BoundingBoxItem 
from .annotator_view import AnnotationView 
from .inference import YOLOInferenceRunner 
from .box_downloader import BoxNavigator


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Place helper functions BEFORE the class that uses them
def load_yolo_to_pixel(yolo_dir, frame_index, frame_width, frame_height):
    """Reads YOLO .txt for one frame and converts to pixel coordinates."""
    filename = os.path.join(yolo_dir, f"{frame_index:05d}.txt") 
    annotations = [] # Changed from 'annotation' to plural for clarity

    if not os.path.exists(filename):
        return annotations

    with open(filename, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                # 1. Parse normalized values
                class_id = int(parts[0])
                xc, yc, w_norm, h_norm = [float(p) for p in parts[1:]]
                    
                # 2. Denormalize dimensions
                w_pixel = w_norm * frame_width
                h_pixel = h_norm * frame_height
                    
                # 3. Denormalize center
                xc_pixel = xc * frame_width
                yc_pixel = yc * frame_height
                    
                # 4. Calculate corners
                x_min = int(xc_pixel - w_pixel / 2)
                y_min = int(yc_pixel - h_pixel / 2)
                x_max = int(xc_pixel + w_pixel / 2)
                y_max = int(yc_pixel + h_pixel / 2)
                    
                # Store as (class_id, x_min, y_min, x_max, y_max)
                annotations.append((class_id, x_min, y_min, x_max, y_max))
        return annotations


def resolve_video_path(local_video_path: str, box_video_name: str) -> str:
    """
    Returns an absolute path to the video to annotate. If a local path is provided,
    it is validated; otherwise the video is downloaded from Box.
    """
    if local_video_path:
        absolute_path = os.path.abspath(os.path.join(PROJECT_ROOT, local_video_path))
        if not os.path.exists(absolute_path):
            raise FileNotFoundError(f"Video file not found: {absolute_path}")
        return absolute_path

    if not box_video_name:
        raise ValueError("Either --video_path or --box_video is required.")

    navigator = BoxNavigator(base_dir=OUTPUT_DIR)
    downloaded_path = navigator.download_vid(box_video_name)
    if not downloaded_path:
        raise FileNotFoundError(f"Could not download '{box_video_name}' from Box.")
    return downloaded_path


class MainWindow(QMainWindow):
    def __init__(self, video_path, yolo_weights):
        super().__init__()
        self.setWindowTitle("YOLO Video Annotator")
        self.setFocusPolicy(Qt.StrongFocus)

        # --- 1. Video and YOLO Setup ---
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            print(f'Error loading video file: {video_path}')
            sys.exit(1)
            
        # Run YOLO Inference (Requires a temporary output folder)
        TEMP_YOLO_OUTPUT = os.path.join(OUTPUT_DIR, "yolo_temp_labels")
        yolo_runner = YOLOInferenceRunner(yolo_weights)
        
        # FIX 1: Provide a temporary output path. self.yolo_dir is assigned the returned path.
        self.yolo_dir = yolo_runner.run_inference_and_save(self.video_path, TEMP_YOLO_OUTPUT)
        
        # Get frame properties
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.current_frame_index = 0
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Load annotations from disk
        self.annotations = [[] for i in range(self.total_frames)]
        for i in range(self.total_frames):
            self.annotations[i] = load_yolo_to_pixel(self.yolo_dir, i, self.frame_width, self.frame_height)
            
        # --- 2. UI Setup ---
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        self.layout = QVBoxLayout(central_widget)
        
        # QGraphicsView Setup
        self.scene = QGraphicsScene()
        self.view = AnnotationView(self.scene)
        self.view.setMinimumSize(640, 480) # Set min size on the View
        self.view.setStyleSheet("background-color: darkgray;")
        self.view.new_box_drawn.connect(self.add_new_annotation)
        self.view.navigate_next.connect(self.go_to_next_frame)
        self.view.navigate_prev.connect(self.go_to_previous_frame)
        self.view.delete_selected.connect(self.delete_selected_boxes)
        
        self.pixmap_item = QGraphicsPixmapItem()
        self.scene.addItem(self.pixmap_item)

        # FIX 2: Add the QGraphicsView widget to the layout, remove QLabel
        self.layout.addWidget(self.view)

        # Box count display
        self.box_count_label = QLabel("Boxes in frame: 0")
        self.layout.addWidget(self.box_count_label)

        # Control Buttons
        self.btn_next = QPushButton("Next Frame >>")
        self.btn_prev = QPushButton('<< Previous Frame')
        self.btn_save = QPushButton('Save dataset')
        self.layout.addWidget(self.btn_next)
        self.layout.addWidget(self.btn_prev)
        self.layout.addWidget(self.btn_save)
        
        # --- 3. Connections and Initial Display ---
        self.btn_next.clicked.connect(self.go_to_next_frame)
        self.btn_prev.clicked.connect(self.go_to_previous_frame)
        self.btn_save.clicked.connect(self.save_dataset)
        self.display_frame()

    def save_current_annotations(self):
        # Clear the old list for the current frame
        self.annotations[self.current_frame_index] = []

        for item in self.scene.items():
            # Check if the item is one of our custom bounding boxes
            if isinstance(item, BoundingBoxItem):
                rect = item.rect()
                
                box_data = (
                    item.class_id,
                    int(rect.left()),
                    int(rect.top()),
                    int(rect.right()),
                    int(rect.bottom())
                )
                # FIX 3: Use self.annotations
                self.annotations[self.current_frame_index].append(box_data)
    
    def add_new_annotation(self, rect: QRectF):
        """Creates a permanent box item and adds data to annotations list."""
        
        # 1. Create the permanent graphical item
        box_item = BoundingBoxItem(rect, class_id=0) 
        self.scene.addItem(box_item)
        
        # 2. Add the pixel coordinates to the current frame's data structure
        new_box_data = (
            box_item.class_id,
            int(rect.left()),
            int(rect.top()),
            int(rect.right()),
            int(rect.bottom())
        )
        self.annotations[self.current_frame_index].append(new_box_data)
        
        print(f"Added new box to frame {self.current_frame_index}")
        self.update_box_count()

    def display_frame(self):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_index)
        ret, frame = self.cap.read()
        
        if ret:
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)

            # Update the QGraphicsPixmapItem
            self.pixmap_item.setPixmap(pixmap)
            # Ensure the view fits the image
            self.view.fitInView(self.pixmap_item, Qt.KeepAspectRatio) 
        else:
            print("Error reading frame.")
        
        # --- Bounding Box Drawing Logic ---
        # Clear ALL old bounding boxes from the scene
        items_to_remove = [item for item in self.scene.items() if isinstance(item, BoundingBoxItem)]
        for item in items_to_remove:
            self.scene.removeItem(item)

        # Draw the bounding boxes for the current frame
        boxes = self.annotations[self.current_frame_index]
        for class_id, x_min, y_min, x_max, y_max in boxes:
            rect_qt = QRectF(x_min, y_min, x_max - x_min, y_max - y_min)
            box_item = BoundingBoxItem(rect_qt, class_id)
            self.scene.addItem(box_item)
        self.update_box_count()
                 
    def go_to_next_frame(self):
        self.save_current_annotations()
        if self.current_frame_index < self.total_frames - 1:
            self.current_frame_index += 1 
            self.display_frame()
            print(f"Showing frame: {self.current_frame_index}")
        else:
            print("End of video reached.")
    
    def go_to_previous_frame(self):
        self.save_current_annotations()
        if self.current_frame_index > 0:
            self.current_frame_index -= 1
            self.display_frame()
            print(f"Showing frame: {self.current_frame_index}")
        else:
            print('No previous frame')
    
    def delete_selected_boxes(self):
        """Remove selected bounding boxes from the scene and stored annotations."""
        selected_items = list(self.scene.selectedItems())
        for item in selected_items:
            if isinstance(item, BoundingBoxItem):
                self.scene.removeItem(item)
        self.save_current_annotations()
        self.display_frame()

    def save_dataset(self):
        """Export frames and labels in YOLO format to outputs/train_data/<video_name> directory."""
        self.save_current_annotations()
        video_stem = os.path.splitext(os.path.basename(self.video_path))[0]
        output_dir = os.path.join(OUTPUT_DIR, "train_data", video_stem)
        images_dir = os.path.join(output_dir, "images")
        labels_dir = os.path.join(output_dir, "labels")
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)

        export_cap = cv2.VideoCapture(self.video_path)
        if not export_cap.isOpened():
            print(f"Could not reopen video for export: {self.video_path}")
            return

        for idx in range(self.total_frames):
            ret, frame = export_cap.read()
            if not ret:
                print(f"Stopped early at frame {idx} while exporting.")
                break

            image_path = os.path.join(images_dir, f"{idx:05d}.jpg")
            cv2.imwrite(image_path, frame)

            label_path = os.path.join(labels_dir, f"{idx:05d}.txt")
            lines = []
            for _, x_min, y_min, x_max, y_max in self.annotations[idx]:
                xc = ((x_min + x_max) / 2) / self.frame_width
                yc = ((y_min + y_max) / 2) / self.frame_height
                w = max((x_max - x_min), 1) / self.frame_width
                h = max((y_max - y_min), 1) / self.frame_height
                # All boxes are class 0 per requirements
                lines.append(f"0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

            with open(label_path, "w", encoding="utf-8") as f:
                f.writelines(lines)

        export_cap.release()
        print(f"Saved dataset to {output_dir}")

    def keyPressEvent(self, event):        
        if event.key() == Qt.Key_Right:
            self.go_to_next_frame()
            
        elif event.key() == Qt.Key_Left: # Use elif to ensure only one key is processed
            self.go_to_previous_frame()
        elif event.key() in (Qt.Key_Delete, Qt.Key_Backspace):
            self.delete_selected_boxes()

        super().keyPressEvent(event)

    def update_box_count(self):
        """Refresh the displayed count of boxes in the current frame."""
        count = len(self.annotations[self.current_frame_index])
        self.box_count_label.setText(f"Boxes in frame: {count}")

def main():
    parser = argparse.ArgumentParser(description="YOLO Video Annotation Tool.")
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        '--video_path', 
        type=str, 
        help='The path to the local MP4 video file to annotate.'
    )
    source_group.add_argument(
        '--box_video',
        type=str,
        help='Exact Box filename to download and annotate (e.g., video.mp4).'
    )
    # Added argument for YOLO weights
    parser.add_argument(
        '--yolo_weights', 
        type=str, 
        required=True, 
        help='The path to the YOLO model weights file (.pt).'
    )
    args = parser.parse_args()
    app = QApplication(sys.argv)
    
    try:
        resolved_video_path = resolve_video_path(args.video_path, args.box_video)
    except (FileNotFoundError, ValueError) as exc:
        print(exc)
        sys.exit(1)

    window = MainWindow(video_path=resolved_video_path, yolo_weights=args.yolo_weights) 
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
