import sys
from PyQt5.QtWidgets import (
    QGraphicsPixmapItem,
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QGraphicsView,
    QGraphicsScene,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QSlider,
    QScrollArea,
    QSplitter,
)
from PyQt5.QtGui import QImage, QPixmap, QColor
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
    """Reads YOLO .txt for one frame and converts to pixel coordinates.
    Supports both old format (5 values) and new format with object_id (6 values).
    For old format, assigns sequential object_ids starting from 0.
    """
    filename = os.path.join(yolo_dir, f"{frame_index:05d}.txt") 
    annotations = [] # Changed from 'annotation' to plural for clarity

    if not os.path.exists(filename):
        return annotations

    sequential_id = 0  # For assigning IDs to old format boxes
    with open(filename, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 6:
                # New format: object_id class_id x_c y_c w h
                object_id = int(parts[0])
                class_id = int(parts[1])
                xc, yc, w_norm, h_norm = [float(p) for p in parts[2:]]
            elif len(parts) == 5:
                # Old format: class_id x_c y_c w h (backward compatibility)
                # Assign sequential object_id for backward compatibility
                object_id = sequential_id
                sequential_id += 1
                class_id = int(parts[0])
                xc, yc, w_norm, h_norm = [float(p) for p in parts[1:]]
            else:
                continue
                    
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
                
            # Store as (object_id, class_id, x_min, y_min, x_max, y_max)
            annotations.append((object_id, class_id, x_min, y_min, x_max, y_max))
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
        self.next_object_id = 0  # For assigning IDs to manually added boxes
        self.all_object_ids = set()  # Track all unique object IDs across all frames
        for i in range(self.total_frames):
            frame_annotations = load_yolo_to_pixel(self.yolo_dir, i, self.frame_width, self.frame_height)
            self.annotations[i] = frame_annotations
            # Track the highest object_id to assign new ones and collect all IDs
            for obj_id, _, _, _, _, _ in frame_annotations:
                if obj_id is not None:
                    self.all_object_ids.add(obj_id)
                    if obj_id >= self.next_object_id:
                        self.next_object_id = obj_id + 1
        
        # Selected object IDs for highlighting and merging
        self.selected_object_ids = set()
            
        # --- 2. UI Setup ---
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Use a splitter to make the UI more compact and scrollable
        splitter = QSplitter(Qt.Vertical)
        main_layout.addWidget(splitter)
        
        # Top section: Objects list and video view
        top_widget = QWidget()
        self.layout = QVBoxLayout(top_widget)
        
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

        # Object list display showing color, object ID, and class
        objects_label = QLabel("Objects in Current Frame:")
        self.layout.addWidget(objects_label)
        self.objects_list = QListWidget()
        self.objects_list.setMaximumHeight(100)
        self.objects_list.setStyleSheet("font-family: monospace; font-size: 10pt;")
        self.objects_list.itemSelectionChanged.connect(self.on_current_frame_object_selected)
        self.layout.addWidget(self.objects_list)

        # Place the view below the list
        self.layout.addWidget(self.view)
        
        # Add top section to splitter
        splitter.addWidget(top_widget)
        
        # Bottom section: Object ID selection and duplicate frames (scrollable)
        bottom_widget = QWidget()
        bottom_layout = QVBoxLayout(bottom_widget)
        bottom_layout.setContentsMargins(3, 3, 3, 3)
        bottom_layout.setSpacing(3)
        
        # Object ID selection menu for merging
        selection_label = QLabel("Select Object IDs to Merge (Ctrl+Click):")
        selection_label.setStyleSheet("font-size: 9pt;")
        bottom_layout.addWidget(selection_label)
        self.object_id_menu = QListWidget()
        self.object_id_menu.setSelectionMode(QListWidget.MultiSelection)
        self.object_id_menu.setMaximumHeight(80)
        self.object_id_menu.setStyleSheet("font-family: monospace; font-size: 8pt;")
        # Populate with all object IDs
        for obj_id in sorted(self.all_object_ids):
            from .bounding_box import generate_color_for_id
            r, g, b = generate_color_for_id(obj_id)
            color_hex = f"#{r:02x}{g:02x}{b:02x}"
            item = QListWidgetItem(f"  ■ {color_hex:8s}  ID: {obj_id}")
            item.setData(Qt.UserRole, obj_id)  # Store object ID in item data
            item.setForeground(QColor(r, g, b))
            self.object_id_menu.addItem(item)
        self.object_id_menu.itemSelectionChanged.connect(self.on_object_ids_selected)
        bottom_layout.addWidget(self.object_id_menu)
        
        # Merge and Delete buttons (compact)
        button_layout = QHBoxLayout()
        self.merge_button = QPushButton("Merge Selected")
        self.merge_button.setEnabled(False)
        self.merge_button.clicked.connect(self.merge_selected_objects)
        button_layout.addWidget(self.merge_button)
        
        self.delete_object_button = QPushButton("Delete from Frame")
        self.delete_object_button.setEnabled(False)
        self.delete_object_button.clicked.connect(self.delete_selected_objects_from_frame)
        button_layout.addWidget(self.delete_object_button)
        bottom_layout.addLayout(button_layout)
        
        # Menu showing frames with duplicate object IDs
        duplicate_label = QLabel("Frames with Duplicate Object IDs:")
        duplicate_label.setStyleSheet("font-size: 9pt;")
        bottom_layout.addWidget(duplicate_label)
        self.duplicate_frames_list = QListWidget()
        self.duplicate_frames_list.setMaximumHeight(80)
        self.duplicate_frames_list.setStyleSheet("font-family: monospace; font-size: 8pt;")
        self.duplicate_frames_list.itemDoubleClicked.connect(self.on_duplicate_frame_clicked)
        bottom_layout.addWidget(self.duplicate_frames_list)
        
        # Make bottom widget scrollable
        scroll_area = QScrollArea()
        scroll_area.setWidget(bottom_widget)
        scroll_area.setWidgetResizable(True)
        scroll_area.setMaximumHeight(220)
        scroll_area.setMinimumHeight(200)
        
        # Add bottom section to splitter
        splitter.addWidget(scroll_area)
        
        # Set splitter sizes (give more space to top section) - reduced bottom size
        splitter.setSizes([750, 220])
        slider_layout = QHBoxLayout()
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(max(0, self.total_frames - 1))
        self.frame_slider.setValue(0)
        self.frame_slider.setTickPosition(QSlider.TicksBelow)
        self.frame_slider.setTickInterval(max(1, self.total_frames // 20))  # ~20 ticks
        self.frame_slider.valueChanged.connect(self.on_slider_changed)
        slider_layout.addWidget(QLabel("Frame:"))
        slider_layout.addWidget(self.frame_slider)
        self.frame_number_label = QLabel("0")
        self.frame_number_label.setMinimumWidth(60)
        slider_layout.addWidget(self.frame_number_label)
        slider_widget = QWidget()
        slider_widget.setLayout(slider_layout)
        self.layout.addWidget(slider_widget)

        # Control Buttons
        button_layout = QHBoxLayout()
        self.btn_prev = QPushButton('<< Previous Frame')
        self.btn_next = QPushButton("Next Frame >>")
        self.btn_save = QPushButton('Save dataset')
        button_layout.addWidget(self.btn_prev)
        button_layout.addWidget(self.btn_next)
        button_layout.addWidget(self.btn_save)
        button_widget = QWidget()
        button_widget.setLayout(button_layout)
        self.layout.addWidget(button_widget)
        
        # Initialize duplicate frames list
        self.update_duplicate_frames_list()
        
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
                    item.object_id,
                    item.class_id,
                    int(rect.left()),
                    int(rect.top()),
                    int(rect.right()),
                    int(rect.bottom())
                )
                # Store as (object_id, class_id, x_min, y_min, x_max, y_max)
                self.annotations[self.current_frame_index].append(box_data)
    
    def add_new_annotation(self, rect: QRectF):
        """Creates a permanent box item and adds data to annotations list."""
        
        # 1. Assign a new object_id for manually added boxes
        new_object_id = self.next_object_id
        self.next_object_id += 1
        
        # 2. Create the permanent graphical item with object_id
        box_item = BoundingBoxItem(rect, class_id=0, object_id=new_object_id) 
        self.scene.addItem(box_item)
        
        # 3. Add the pixel coordinates to the current frame's data structure
        new_box_data = (
            box_item.object_id,
            box_item.class_id,
            int(rect.left()),
            int(rect.top()),
            int(rect.right()),
            int(rect.bottom())
        )
        self.annotations[self.current_frame_index].append(new_box_data)
        
        print(f"Added new box (object_id={new_object_id}) to frame {self.current_frame_index}")
        self.update_box_count()

    def display_frame(self):
        # Update slider position (block signals to avoid recursion)
        self.frame_slider.blockSignals(True)
        self.frame_slider.setValue(self.current_frame_index)
        self.frame_slider.blockSignals(False)
        
        # Update frame number label
        self.frame_number_label.setText(f"{self.current_frame_index}/{self.total_frames-1}")
        
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
            # Allow zoom to persist; only auto-fit if user has not zoomed
            self.view.setSceneRect(self.pixmap_item.boundingRect())
            if not self.view.user_zoomed:
                self.view.fit_to_pixmap(self.pixmap_item)
        else:
            print("Error reading frame.")
        
        # --- Bounding Box Drawing Logic ---
        # Clear ALL old bounding boxes from the scene
        items_to_remove = [item for item in self.scene.items() if isinstance(item, BoundingBoxItem)]
        for item in items_to_remove:
            self.scene.removeItem(item)

        # Draw the bounding boxes for the current frame
        boxes = self.annotations[self.current_frame_index]
        for obj_id, class_id, x_min, y_min, x_max, y_max in boxes:
            rect_qt = QRectF(x_min, y_min, x_max - x_min, y_max - y_min)
            # Check if this object ID is selected for highlighting
            is_highlighted = obj_id in self.selected_object_ids
            box_item = BoundingBoxItem(rect_qt, class_id=class_id, object_id=obj_id, is_highlighted=is_highlighted)
            self.scene.addItem(box_item)
        self.update_box_count()
                 
    def on_slider_changed(self, value):
        """Handle slider value change to navigate to a specific frame."""
        # Prevent recursive updates
        if value != self.current_frame_index:
            self.save_current_annotations()
            self.current_frame_index = value
            self.display_frame()
            print(f"Showing frame: {self.current_frame_index}")
                 
    def go_to_next_frame(self):
        self.save_current_annotations()
        if self.current_frame_index < self.total_frames - 1:
            self.current_frame_index += 1 
            # Update slider without triggering valueChanged (to avoid recursion)
            self.frame_slider.blockSignals(True)
            self.frame_slider.setValue(self.current_frame_index)
            self.frame_slider.blockSignals(False)
            self.display_frame()
            print(f"Showing frame: {self.current_frame_index}")
        else:
            print("End of video reached.")
    
    def go_to_previous_frame(self):
        self.save_current_annotations()
        if self.current_frame_index > 0:
            self.current_frame_index -= 1
            # Update slider without triggering valueChanged (to avoid recursion)
            self.frame_slider.blockSignals(True)
            self.frame_slider.setValue(self.current_frame_index)
            self.frame_slider.blockSignals(False)
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
            for obj_id, class_id, x_min, y_min, x_max, y_max in self.annotations[idx]:
                xc = ((x_min + x_max) / 2) / self.frame_width
                yc = ((y_min + y_max) / 2) / self.frame_height
                w = max((x_max - x_min), 1) / self.frame_width
                h = max((y_max - y_min), 1) / self.frame_height
                # All boxes are class 0 per requirements (but we preserve object_id for internal tracking)
                lines.append(f"0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

            with open(label_path, "w", encoding="utf-8") as f:
                f.writelines(lines)

        export_cap.release()
        print(f"Saved dataset to {output_dir}")

    def keyPressEvent(self, event):        
        if event.key() == Qt.Key_Right:
            self.go_to_next_frame()
            # Update slider position
            self.frame_slider.blockSignals(True)
            self.frame_slider.setValue(self.current_frame_index)
            self.frame_slider.blockSignals(False)
            
        elif event.key() == Qt.Key_Left: # Use elif to ensure only one key is processed
            self.go_to_previous_frame()
            # Update slider position
            self.frame_slider.blockSignals(True)
            self.frame_slider.setValue(self.current_frame_index)
            self.frame_slider.blockSignals(False)
        elif event.key() in (Qt.Key_Delete, Qt.Key_Backspace):
            self.delete_selected_boxes()

        super().keyPressEvent(event)

    def update_box_count(self):
        """Refresh the displayed list of objects with their color, object ID, and class."""
        # Block signals to prevent recursive updates
        self.objects_list.blockSignals(True)
        self.objects_list.clear()
        
        # Get all bounding box items from the scene to get their colors
        scene_items = {}
        for item in self.scene.items():
            if isinstance(item, BoundingBoxItem):
                scene_items[item.object_id] = item
        
        # Display each object in the current frame
        boxes = self.annotations[self.current_frame_index]
        for obj_id, class_id, x_min, y_min, x_max, y_max in boxes:
            # Get the color from the scene item if available
            if obj_id in scene_items:
                color = scene_items[obj_id].base_color
                r, g, b = color.red(), color.green(), color.blue()
            else:
                # Generate color if not in scene (shouldn't happen, but safety check)
                from .bounding_box import generate_color_for_id
                r, g, b = generate_color_for_id(obj_id)
            
            # Create list item with color indicator and info
            color_hex = f"#{r:02x}{g:02x}{b:02x}"
            item_text = f"  ■ {color_hex:8s}  Object ID: {obj_id:4d}  Class: {class_id}"
            
            list_item = QListWidgetItem(item_text)
            list_item.setData(Qt.UserRole, obj_id)  # Store object ID for syncing
            # Set text color to match the object color (with some contrast)
            text_color = QColor(r, g, b)
            list_item.setForeground(text_color)
            self.objects_list.addItem(list_item)
            
            # Sync selection if this object ID is selected
            if obj_id in self.selected_object_ids:
                list_item.setSelected(True)
        
        self.objects_list.blockSignals(False)
        
        # Update frame number label
        self.frame_number_label.setText(f"{self.current_frame_index}/{self.total_frames-1}")
        
        # Show count in window title
        count = len(boxes)
        self.setWindowTitle(f"YOLO Video Annotator - Frame {self.current_frame_index}/{self.total_frames-1} - {count} objects")
    
    def on_current_frame_object_selected(self):
        """Handle selection in the current frame objects list - sync with object ID menu."""
        selected_items = self.objects_list.selectedItems()
        selected_ids = {item.data(Qt.UserRole) for item in selected_items if item.data(Qt.UserRole) is not None}
        
        if selected_ids != self.selected_object_ids:
            self.selected_object_ids = selected_ids
            # Sync with object ID menu
            self.sync_object_id_menu_selection()
            # Update button states
            self.merge_button.setEnabled(len(self.selected_object_ids) > 1)
            self.delete_object_button.setEnabled(len(self.selected_object_ids) > 0)
            # Refresh display to show highlights
            self.display_frame()
    
    def sync_object_id_menu_selection(self):
        """Sync selection in object ID menu with selected_object_ids."""
        self.object_id_menu.blockSignals(True)
        # Clear all selections first
        for i in range(self.object_id_menu.count()):
            item = self.object_id_menu.item(i)
            obj_id = item.data(Qt.UserRole)
            item.setSelected(obj_id in self.selected_object_ids)
        self.object_id_menu.blockSignals(False)
    
    def on_object_ids_selected(self):
        """Handle selection changes in the object ID menu."""
        selected_items = self.object_id_menu.selectedItems()
        new_selected_ids = {item.data(Qt.UserRole) for item in selected_items}
        
        if new_selected_ids != self.selected_object_ids:
            self.selected_object_ids = new_selected_ids
            
            # Sync with current frame objects list
            self.sync_current_frame_objects_list()
            
            # Enable merge button only if multiple IDs are selected
            self.merge_button.setEnabled(len(self.selected_object_ids) > 1)
            
            # Enable delete button if at least one ID is selected
            self.delete_object_button.setEnabled(len(self.selected_object_ids) > 0)
            
            # Refresh the current frame to show/hide highlights
            self.display_frame()
    
    def sync_current_frame_objects_list(self):
        """Sync selection in current frame objects list with selected_object_ids."""
        self.objects_list.blockSignals(True)
        for i in range(self.objects_list.count()):
            item = self.objects_list.item(i)
            obj_id = item.data(Qt.UserRole)
            if obj_id is not None:
                item.setSelected(obj_id in self.selected_object_ids)
        self.objects_list.blockSignals(False)
    
    def merge_selected_objects(self):
        """Merge selected object IDs into the first selected ID."""
        if len(self.selected_object_ids) < 2:
            return
        
        # Get sorted list - first one is the target ID
        sorted_ids = sorted(self.selected_object_ids)
        target_id = sorted_ids[0]
        ids_to_merge = sorted_ids[1:]
        
        print(f"Merging object IDs {ids_to_merge} into {target_id}")
        
        # Update all annotations across all frames
        for frame_idx in range(self.total_frames):
            for i, (obj_id, class_id, x_min, y_min, x_max, y_max) in enumerate(self.annotations[frame_idx]):
                if obj_id in ids_to_merge:
                    # Replace with target ID
                    self.annotations[frame_idx][i] = (target_id, class_id, x_min, y_min, x_max, y_max)
        
        # Update all_object_ids set
        self.all_object_ids.difference_update(ids_to_merge)
        
        # Clear selection
        self.object_id_menu.clearSelection()
        self.selected_object_ids.clear()
        self.merge_button.setEnabled(False)
        
        # Refresh the object ID menu
        self.object_id_menu.clear()
        for obj_id in sorted(self.all_object_ids):
            from .bounding_box import generate_color_for_id
            r, g, b = generate_color_for_id(obj_id)
            color_hex = f"#{r:02x}{g:02x}{b:02x}"
            item = QListWidgetItem(f"  ■ {color_hex:8s}  Object ID: {obj_id}")
            item.setData(Qt.UserRole, obj_id)
            item.setForeground(QColor(r, g, b))
            self.object_id_menu.addItem(item)
        
        # Refresh current frame display
        self.display_frame()
        
        print(f"Merge complete. All objects now use ID {target_id}")
    
    def delete_selected_objects_from_frame(self):
        """Delete all bounding boxes with selected object IDs from the current frame only."""
        if len(self.selected_object_ids) == 0:
            return
        
        # Filter out annotations with selected object IDs
        original_count = len(self.annotations[self.current_frame_index])
        self.annotations[self.current_frame_index] = [
            (obj_id, class_id, x_min, y_min, x_max, y_max)
            for obj_id, class_id, x_min, y_min, x_max, y_max in self.annotations[self.current_frame_index]
            if obj_id not in self.selected_object_ids
        ]
        deleted_count = original_count - len(self.annotations[self.current_frame_index])
        
        print(f"Deleted {deleted_count} bounding box(es) with object ID(s) {self.selected_object_ids} from frame {self.current_frame_index}")
        
        # Refresh the display
        self.display_frame()
        
        # Update duplicate frames list after deletion
        self.update_duplicate_frames_list()
        
        # Update duplicate frames list after deletion
        self.update_duplicate_frames_list()
    
    def update_duplicate_frames_list(self):
        """Find and display frames that have multiple boxes with the same object ID."""
        self.duplicate_frames_list.clear()
        
        duplicate_frames = []
        for frame_idx in range(self.total_frames):
            # Count occurrences of each object ID in this frame
            object_id_counts = {}
            for obj_id, _, _, _, _, _ in self.annotations[frame_idx]:
                if obj_id is not None:
                    object_id_counts[obj_id] = object_id_counts.get(obj_id, 0) + 1
            
            # Check if any object ID appears more than once
            for obj_id, count in object_id_counts.items():
                if count > 1:
                    duplicate_frames.append((frame_idx, obj_id, count))
                    break  # Only add frame once even if multiple IDs have duplicates
        
        # Sort by frame number
        duplicate_frames.sort(key=lambda x: x[0])
        
        # Add to list widget
        for frame_idx, obj_id, count in duplicate_frames:
            item_text = f"Frame {frame_idx:5d} - Object ID {obj_id} appears {count} times"
            item = QListWidgetItem(item_text)
            item.setData(Qt.UserRole, frame_idx)  # Store frame number
            self.duplicate_frames_list.addItem(item)
        
        if len(duplicate_frames) == 0:
            no_duplicates_item = QListWidgetItem("No frames with duplicate object IDs")
            no_duplicates_item.setFlags(Qt.NoItemFlags)  # Make it non-selectable
            self.duplicate_frames_list.addItem(no_duplicates_item)
    
    def on_duplicate_frame_clicked(self, item):
        """Navigate to the frame when a duplicate frame item is double-clicked."""
        frame_idx = item.data(Qt.UserRole)
        if frame_idx is not None:
            self.save_current_annotations()
            self.current_frame_index = frame_idx
            # Update slider
            self.frame_slider.blockSignals(True)
            self.frame_slider.setValue(self.current_frame_index)
            self.frame_slider.blockSignals(False)
            self.display_frame()
            print(f"Navigated to frame {frame_idx}")

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
