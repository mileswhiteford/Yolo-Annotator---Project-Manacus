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
    QSizePolicy,
    QFrame,
)
from PyQt5.QtGui import QImage, QPixmap, QColor, QPen, QBrush, QFont, QPolygonF, QPalette, QFontMetrics
from PyQt5.QtCore import Qt, QRectF, pyqtSignal, QPointF, QPoint
import cv2
import argparse
import os
from .bounding_box import BoundingBoxItem, generate_color_for_id
from .annotator_view import AnnotationView 
from .inference import YOLOInferenceRunner 
from .box_downloader import BoxNavigator


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

class NoWheelScrollArea(QScrollArea):
    """Scroll area that ignores wheel events to keep scrolling exclusive to timeline."""
    def wheelEvent(self, event):
        event.ignore()

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

class ObjectTimelineView(QGraphicsView):
    """Simple timeline showing per-object presence across frames."""
    frame_selected = pyqtSignal(int)
    objects_selected = pyqtSignal(list)
    segments_selected = pyqtSignal(list)  # list of (obj_id, start_frame, end_frame)
    segments_delete_requested = pyqtSignal()

    def __init__(self, total_frames: int, parent=None):
        self.scene_obj = QGraphicsScene(parent)
        super().__init__(self.scene_obj, parent)
        self.total_frames = max(total_frames, 1)
        # Fixed margin sized for up to 3-digit IDs to avoid jitter
        fm = QFontMetrics(QFont())
        self.left_margin = max(70, fm.boundingRect("ID 999").width() + 12)
        self.base_width = 800
        self.timeline_width = self.base_width
        self.bar_height = 10
        self.row_spacing = 8
        self.setMinimumHeight(180)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setFrameStyle(QFrame.NoFrame)
        self.setFocusPolicy(Qt.StrongFocus)
        self.bg_color = self.palette().color(QPalette.Base)
        self.presence = {}
        self.duplicate_presence = {}
        self.zoom_factor = 1.0
        self.zoom_min = 1.0  # do not zoom out past fully visible timeline
        self.zoom_max = 5.0
        self.current_frame = 0
        self.dragging_marker = False
        self.panning = False
        self._last_pan_pos = None
        self.scrubbing = False
        self.selected_object_ids = set()
        self.bar_geometries = []
        self.label_items = []
        self.label_geometries = []
        self.label_column_bg = None
        self.marker_handle_rect = None
        self.marker_bar_rect = None
        self.selecting = False
        self.selection_start = None
        self.selection_rect_item = None
        self.selected_segments = []
        self._redraw()

    def set_total_frames(self, total_frames: int):
        self.total_frames = max(total_frames, 1)
        self._redraw()

    def set_presence(self, presence: dict):
        """presence: {object_id: set/list of frame indices}"""
        self.presence = {k: set(v) for k, v in presence.items()}
        self._redraw()

    def set_duplicates(self, duplicates: dict):
        """duplicates: {object_id: set of frame indices with duplicate detections}"""
        self.duplicate_presence = {k: set(v) for k, v in duplicates.items()}
        self._redraw()

    def set_current_frame(self, frame_idx: int):
        self.current_frame = max(0, min(frame_idx, self.total_frames - 1))
        self._redraw()

    def resizeEvent(self, event):
        self.base_width = max(200, self.viewport().width() - self.left_margin - 20)
        self.timeline_width = max(200, self.base_width * self.zoom_factor)
        self._redraw()
        super().resizeEvent(event)

    def _redraw(self):
        self.scene_obj.clear()
        self.bar_geometries = []
        # Minimal fixed margin; IDs not rendered
        self.left_margin = 12
        self.base_width = max(200, self.viewport().width() - self.left_margin - 20)
        self.timeline_width = max(200, self.base_width * self.zoom_factor)
        scale = self.timeline_width / float(max(self.total_frames - 1, 1))
        y = 10
        obj_rows = {}
        row_idx = 0
        for obj_id in sorted(self.presence.keys()):
            frames = sorted(self.presence[obj_id])
            if not frames:
                continue
            r, g, b = generate_color_for_id(obj_id)
            color = QColor(r, g, b)
            is_selected = obj_id in self.selected_object_ids
            # Muted, softer colors
            base_color = color.lighter(120)
            pen_color = base_color.lighter(120) if is_selected else base_color
            pen = QPen(pen_color, 2 if is_selected else 1)
            brush = QBrush(base_color.lighter(130) if is_selected else base_color)
            obj_rows[obj_id] = row_idx

            start = frames[0]
            end = frames[0]
            for f in frames[1:] + [None]:
                if f is not None and f == end + 1:
                    end = f
                    continue
                x_start = self.left_margin + start * scale
                width = max(2, (end - start + 1) * scale)
                rect = QRectF(x_start, y, width, self.bar_height)
                self.scene_obj.addRect(rect, pen, brush)
                self.bar_geometries.append((obj_id, rect))
                dup_frames = sorted(self.duplicate_presence.get(obj_id, set()))
                dup_segment = []
                for df in dup_frames:
                    if df < start or df > end:
                        continue
                    if not dup_segment:
                        dup_segment = [df, df]
                    elif df == dup_segment[1] + 1:
                        dup_segment[1] = df
                    else:
                        seg_x = self.left_margin + dup_segment[0] * scale
                        seg_w = max(2, (dup_segment[1] - dup_segment[0] + 1) * scale)
                        line_rect = QRectF(seg_x, y + self.bar_height / 2 - 1, seg_w, 2)
                        self.scene_obj.addRect(line_rect, QPen(Qt.black, 1), QBrush(Qt.black))
                        dup_segment = [df, df]
                if dup_segment:
                    seg_x = self.left_margin + dup_segment[0] * scale
                    seg_w = max(2, (dup_segment[1] - dup_segment[0] + 1) * scale)
                    line_rect = QRectF(seg_x, y + self.bar_height / 2 - 1, seg_w, 2)
                    self.scene_obj.addRect(line_rect, QPen(Qt.black, 1), QBrush(Qt.black))
                start = f
                end = f
            y += self.bar_height + self.row_spacing
            row_idx += 1

        total_height = y + 20
        marker_x = self.left_margin + self.current_frame * scale
        marker_pen = QPen(QColor(90, 90, 90, 180), 1.5, Qt.SolidLine)
        self.scene_obj.addLine(marker_x, 0, marker_x, total_height, marker_pen)
        handle_width = 10
        handle_height = 16
        self.marker_handle_rect = QRectF(marker_x - handle_width / 2, 0, handle_width, total_height)
        handle_brush = QBrush(QColor(140, 140, 140, 60))
        self.scene_obj.addRect(self.marker_handle_rect, QPen(Qt.NoPen), handle_brush)
        bar_height = 6
        self.marker_bar_rect = QRectF(self.left_margin, 0, self.timeline_width, bar_height)
        bar_pen = QPen(QColor(110, 110, 110, 160), 1.2)
        bar_brush = QBrush(QColor(170, 170, 170, 120))
        self.scene_obj.addRect(self.marker_bar_rect, bar_pen, bar_brush)

        if self.selected_segments:
            overlay_brush = QBrush(QColor(0, 0, 0, 60))
            overlay_pen = QPen(QColor(0, 0, 0, 120), 1, Qt.DashLine)
            for obj_id, s_frame, e_frame in self.selected_segments:
                if obj_id not in obj_rows:
                    continue
                row_idx = obj_rows[obj_id]
                y_pos = 10 + row_idx * (self.bar_height + self.row_spacing)
                x_start = self.left_margin + s_frame * scale
                width = max(2, (e_frame - s_frame + 1) * scale)
                sel_rect = QRectF(x_start, y_pos, width, self.bar_height)
                self.scene_obj.addRect(sel_rect, overlay_pen, overlay_brush)

        self.scene_obj.setSceneRect(0, 0, self.left_margin + self.timeline_width, total_height + self.bar_height + 10)

    def _frame_from_x(self, x: float) -> int:
        scale = self.timeline_width / float(max(self.total_frames - 1, 1))
        frame = int(round((x - self.left_margin) / scale))
        return max(0, min(frame, self.total_frames - 1))

    def zoom_in(self):
        new_zoom = max(self.zoom_min, min(self.zoom_max, self.zoom_factor * 1.1))
        if abs(new_zoom - self.zoom_factor) > 1e-3:
            self.zoom_factor = new_zoom
            self.timeline_width = max(200, self.base_width * self.zoom_factor)
            self._redraw()

    def zoom_out(self):
        new_zoom = max(self.zoom_min, min(self.zoom_max, self.zoom_factor / 1.1))
        if abs(new_zoom - self.zoom_factor) > 1e-3:
            self.zoom_factor = new_zoom
            self.timeline_width = max(200, self.base_width * self.zoom_factor)
            self._redraw()

    def _object_id_at(self, pos) -> int:
        """Return object_id for the bar under the given scene position, else None."""
        for obj_id, rect in self.bar_geometries:
            if rect.contains(pos):
                return obj_id
        return None

    def _scene_left(self) -> float:
        """Scene x-coordinate at the left edge of the viewport."""
        return self.mapToScene(QPoint(0, 0)).x()

    def scrollContentsBy(self, dx, dy):
        super().scrollContentsBy(dx, dy)

    def set_selected_object_ids(self, obj_ids, emit_signal=False):
        """Update selected IDs shown on the timeline."""
        new_set = set(obj_ids)
        if new_set == self.selected_object_ids:
            return
        self.selected_object_ids = new_set
        self._redraw()
        if emit_signal:
            self.objects_selected.emit(sorted(self.selected_object_ids))

    def clear_selected_segments(self, emit_signal=True):
        self.selected_segments = []
        self._redraw()
        if emit_signal:
            self.segments_selected.emit([])

    def _apply_selection_rect(self, rect: QRectF):
        if rect.isNull() or rect.width() == 0 or rect.height() == 0:
            self.clear_selected_segments()
            return
        ranges = {}
        for obj_id, bar_rect in self.bar_geometries:
            if not rect.intersects(bar_rect):
                continue
            inter = rect.intersected(bar_rect)
            start_f = self._frame_from_x(inter.left())
            end_f = self._frame_from_x(inter.right())
            if end_f < start_f:
                start_f, end_f = end_f, start_f
            ranges.setdefault(obj_id, []).append((start_f, end_f))
        merged_segments = []
        for obj_id, segs in ranges.items():
            segs.sort()
            cur_s, cur_e = segs[0]
            for s, e in segs[1:]:
                if s <= cur_e + 1:
                    cur_e = max(cur_e, e)
                else:
                    merged_segments.append((obj_id, cur_s, cur_e))
                    cur_s, cur_e = s, e
            merged_segments.append((obj_id, cur_s, cur_e))
        self.selected_segments = merged_segments
        self._redraw()
        self.segments_selected.emit(self.selected_segments)

    def mousePressEvent(self, event):
        scene_pos = self.mapToScene(event.pos())
        x = scene_pos.x()
        marker_x = self.left_margin + (self.current_frame / float(max(self.total_frames - 1, 1))) * self.timeline_width
        if event.button() in (Qt.RightButton, Qt.MiddleButton):
            self.setFocus()
            self.panning = True
            self._last_pan_pos = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
            return
        if self.marker_handle_rect and self.marker_handle_rect.contains(scene_pos) and event.button() == Qt.LeftButton:
            self.setFocus()
            self.dragging_marker = True
            event.accept()
            return
        if self.marker_bar_rect and self.marker_bar_rect.contains(scene_pos) and event.button() == Qt.LeftButton:
            self.setFocus()
            frame = self._frame_from_x(x)
            self.set_current_frame(frame)
            self.frame_selected.emit(frame)
            self.dragging_marker = True
            event.accept()
            return
        clicked_obj_id = self._object_id_at(scene_pos)
        if clicked_obj_id is not None and event.button() == Qt.LeftButton:
            self.setFocus()
            modifiers = event.modifiers()
            new_selection = set(self.selected_object_ids)
            if modifiers & (Qt.ShiftModifier | Qt.ControlModifier | Qt.MetaModifier):
                if clicked_obj_id in new_selection:
                    new_selection.remove(clicked_obj_id)
                else:
                    new_selection.add(clicked_obj_id)
            else:
                new_selection = {clicked_obj_id}
            self.set_selected_object_ids(new_selection, emit_signal=True)
            # Start selection box for range selection
            self.selecting = True
            self.selection_start = scene_pos
            if self.selection_rect_item:
                self.scene_obj.removeItem(self.selection_rect_item)
            self.selection_rect_item = self.scene_obj.addRect(QRectF(scene_pos, scene_pos), QPen(Qt.black, 1, Qt.DashLine), QBrush(QColor(0, 0, 0, 30)))
            event.accept()
            return
        if x >= self.left_margin and event.button() == Qt.LeftButton:
            self.setFocus()
            if clicked_obj_id is None:
                self.set_selected_object_ids(set(), emit_signal=True)
                self.clear_selected_segments()
            self.selecting = True
            self.selection_start = scene_pos
            if self.selection_rect_item:
                self.scene_obj.removeItem(self.selection_rect_item)
            self.selection_rect_item = self.scene_obj.addRect(QRectF(scene_pos, scene_pos), QPen(Qt.black, 1, Qt.DashLine), QBrush(QColor(0, 0, 0, 30)))
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.panning and event.buttons() & (Qt.RightButton | Qt.MiddleButton):
            delta = event.pos() - self._last_pan_pos
            self._last_pan_pos = event.pos()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            event.accept()
            return
        if self.selecting and event.buttons() & Qt.LeftButton and self.selection_rect_item:
            scene_pos = self.mapToScene(event.pos())
            rect = QRectF(self.selection_start, scene_pos).normalized()
            self.selection_rect_item.setRect(rect)
            event.accept()
            return
        if self.dragging_marker and event.buttons() & Qt.LeftButton:
            scene_pos = self.mapToScene(event.pos())
            frame = self._frame_from_x(scene_pos.x())
            if frame != self.current_frame:
                self.set_current_frame(frame)
                self.frame_selected.emit(frame)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.panning and event.button() in (Qt.RightButton, Qt.MiddleButton):
            self.panning = False
            self._last_pan_pos = None
            self.setCursor(Qt.ArrowCursor)
            event.accept()
            return
        if self.dragging_marker and event.button() == Qt.LeftButton:
            self.dragging_marker = False
            self.scrubbing = False
            event.accept()
            return
        if self.selecting and event.button() == Qt.LeftButton:
            self.selecting = False
            if self.selection_rect_item:
                sel_rect = self.selection_rect_item.rect()
                self.scene_obj.removeItem(self.selection_rect_item)
                self.selection_rect_item = None
                self._apply_selection_rect(sel_rect)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event):
        # Wheel/trackpad scroll: vertical for up/down, horizontal for side-to-side
        delta = event.angleDelta()
        handled = False
        if delta.y() != 0:
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            handled = True
        if delta.x() != 0:
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            handled = True
        if handled:
            event.accept()
        else:
            super().wheelEvent(event)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Left:
            new_frame = max(0, self.current_frame - 1)
            if new_frame != self.current_frame:
                self.set_current_frame(new_frame)
                self.frame_selected.emit(new_frame)
            event.accept()
            return
        if event.key() == Qt.Key_Right:
            new_frame = min(self.total_frames - 1, self.current_frame + 1)
            if new_frame != self.current_frame:
                self.set_current_frame(new_frame)
                self.frame_selected.emit(new_frame)
            event.accept()
            return
        if event.key() in (Qt.Key_Delete, Qt.Key_Backspace):
            self.segments_delete_requested.emit()
            event.accept()
            return
        super().keyPressEvent(event)


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
        outer_scroll = NoWheelScrollArea()
        outer_scroll.setWidgetResizable(True)
        central_widget = QWidget()
        outer_scroll.setWidget(central_widget)
        self.setCentralWidget(outer_scroll)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(6, 6, 6, 6)
        main_layout.setSpacing(6)
        
        # Use a splitter to make the UI more compact and scrollable
        splitter = QSplitter(Qt.Vertical)
        main_layout.addWidget(splitter)
        
        # Top section: Objects list and video view
        top_widget = QWidget()
        self.layout = QVBoxLayout(top_widget)
        
        # QGraphicsView Setup
        self.scene = QGraphicsScene()
        self.view = AnnotationView(self.scene)
        self.view.setMinimumSize(480, 360) # Set min size on the View
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
        self.objects_list.setMaximumHeight(80)
        self.objects_list.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.objects_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
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
        self.delete_id_global_button = QPushButton("Delete Object From Video")
        self.delete_id_global_button.setEnabled(False)
        self.delete_id_global_button.clicked.connect(self.delete_selected_ids_globally)
        button_layout.addWidget(self.delete_id_global_button)
        self.delete_segments_button = QPushButton("Delete Selected Range(s)")
        self.delete_segments_button.setEnabled(False)
        self.delete_segments_button.clicked.connect(self.delete_selected_ranges)
        button_layout.addWidget(self.delete_segments_button)
        bottom_layout.addLayout(button_layout)

        # Timeline view for object presence
        timeline_label = QLabel("Object Timelines (click to jump to frame):")
        timeline_label.setStyleSheet("font-size: 9pt;")
        timeline_header = QHBoxLayout()
        timeline_header.addWidget(timeline_label)
        timeline_header.addStretch(1)
        self.frame_number_label = QLabel("0")
        self.frame_number_label.setAlignment(Qt.AlignCenter)
        self.frame_number_label.setMinimumWidth(90)
        timeline_header.addWidget(self.frame_number_label)
        timeline_header.addStretch(1)
        self.btn_timeline_zoom_out = QPushButton("Zoom -")
        self.btn_timeline_zoom_out.setMaximumWidth(70)
        self.btn_timeline_zoom_in = QPushButton("Zoom +")
        self.btn_timeline_zoom_in.setMaximumWidth(70)
        timeline_header.addWidget(self.btn_timeline_zoom_out)
        timeline_header.addWidget(self.btn_timeline_zoom_in)
        bottom_layout.addLayout(timeline_header)
        self.timeline_view = ObjectTimelineView(self.total_frames)
        bottom_layout.addWidget(self.timeline_view)
        self.timeline_view.frame_selected.connect(self.on_timeline_frame_selected)
        self.timeline_view.objects_selected.connect(self.on_timeline_objects_selected)
        self.timeline_view.segments_selected.connect(self.on_timeline_segments_selected)
        self.timeline_view.segments_delete_requested.connect(self.delete_selected_ranges)
        self.btn_timeline_zoom_in.clicked.connect(self.timeline_view.zoom_in)
        self.btn_timeline_zoom_out.clicked.connect(self.timeline_view.zoom_out)
        self.timeline_selected_segments = []
        
        # Make bottom widget scrollable
        scroll_area = NoWheelScrollArea()
        scroll_area.setWidget(bottom_widget)
        scroll_area.setWidgetResizable(True)
        scroll_area.setMaximumHeight(240)
        scroll_area.setMinimumHeight(140)
        
        # Add bottom section to splitter
        splitter.addWidget(scroll_area)
        
        # Set splitter sizes (give more space to timeline than before)
        splitter.setSizes([680, 320])
        # Keep internal slider for programmatic sync, but do not show in UI
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(max(0, self.total_frames - 1))
        self.frame_slider.setValue(0)
        self.frame_slider.setTickPosition(QSlider.TicksBelow)
        self.frame_slider.setTickInterval(max(1, self.total_frames // 20))  # ~20 ticks
        self.frame_slider.valueChanged.connect(self.on_slider_changed)

        # Control Buttons
        button_layout = QHBoxLayout()
        button_layout.setContentsMargins(0, 0, 0, 0)
        button_layout.setSpacing(6)
        self.btn_prev = QPushButton('<< Previous Frame')
        self.btn_prev.setFixedHeight(28)
        self.btn_next = QPushButton("Next Frame >>")
        self.btn_next.setFixedHeight(28)
        self.btn_save = QPushButton('Save dataset')
        self.btn_save.setFixedHeight(28)
        button_layout.addWidget(self.btn_prev)
        button_layout.addWidget(self.btn_next)
        button_layout.addWidget(self.btn_save)
        button_widget = QWidget()
        button_widget.setLayout(button_layout)
        self.layout.addWidget(button_widget)
        self.data_dirty = True  # Start dirty until first explicit save
        self._update_save_button_label()
        
        # --- 3. Connections and Initial Display ---
        self.btn_next.clicked.connect(self.go_to_next_frame)
        self.btn_prev.clicked.connect(self.go_to_previous_frame)
        self.btn_save.clicked.connect(self.save_dataset)
        self.refresh_timeline()
        self.display_frame()

    def _update_save_button_label(self):
        if self.data_dirty:
            self.btn_save.setText("Save dataset (unsaved)")
            self.btn_save.setStyleSheet("")
        else:
            self.btn_save.setText("Dataset saved")
            self.btn_save.setStyleSheet("color: green; font-weight: bold;")

    def mark_dirty(self):
        if not self.data_dirty:
            self.data_dirty = True
            self._update_save_button_label()

    def mark_clean(self):
        if self.data_dirty:
            self.data_dirty = False
            self._update_save_button_label()

    def _determine_draw_object_id(self):
        """Return the object_id to use when drawing a new box."""
        if self.timeline_selected_segments:
            return self.timeline_selected_segments[0][0]
        if self.selected_object_ids:
            return sorted(self.selected_object_ids)[0]
        return None

    def save_current_annotations(self):
        # Clear the old list for the current frame
        previous = list(self.annotations[self.current_frame_index])
        new_entries = []

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
                new_entries.append(box_data)
        self.annotations[self.current_frame_index] = new_entries
        if previous != new_entries:
            self.mark_dirty()
    
    def add_new_annotation(self, rect: QRectF):
        """Creates a permanent box item and adds data to annotations list."""
        target_obj_id = self._determine_draw_object_id()
        if target_obj_id is None:
            target_obj_id = self.next_object_id
            self.next_object_id += 1
        if target_obj_id not in self.all_object_ids:
            self.all_object_ids.add(target_obj_id)
            if target_obj_id >= self.next_object_id:
                self.next_object_id = target_obj_id + 1

        # Create the permanent graphical item with chosen object_id
        box_item = BoundingBoxItem(rect, class_id=0, object_id=target_obj_id) 
        self.scene.addItem(box_item)
        
        # 3. Add the pixel coordinates to the current frame's data structure
        new_box_data = (
            target_obj_id,
            box_item.class_id,
            int(rect.left()),
            int(rect.top()),
            int(rect.right()),
            int(rect.bottom())
        )
        self.annotations[self.current_frame_index].append(new_box_data)
        
        # Update selection to stick to the target object for subsequent draws
        self.selected_object_ids = {target_obj_id}
        self.timeline_view.set_selected_object_ids(self.selected_object_ids)
        self.timeline_selected_segments = []
        self.delete_segments_button.setEnabled(False)
        self.sync_current_frame_objects_list()
        self.merge_button.setEnabled(False)
        self.delete_object_button.setEnabled(True)
        self.delete_id_global_button.setEnabled(True)
        
        print(f"Added new box (object_id={target_obj_id}) to frame {self.current_frame_index}")
        self.refresh_timeline()
        self.update_box_count()
        self.mark_dirty()

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
        self.timeline_view.set_current_frame(self.current_frame_index)

    def refresh_timeline(self):
        """Rebuild object timeline presence map and refresh the view."""
        presence = {}
        duplicates = {}
        for frame_idx, frame_annotations in enumerate(self.annotations):
            counts = {}
            for obj_id, _, _, _, _, _ in frame_annotations:
                if obj_id is None:
                    continue
                if obj_id not in presence:
                    presence[obj_id] = set()
                presence[obj_id].add(frame_idx)
                counts[obj_id] = counts.get(obj_id, 0) + 1
            for obj_id, count in counts.items():
                if count > 1:
                    duplicates.setdefault(obj_id, set()).add(frame_idx)
        self.timeline_view.set_total_frames(self.total_frames)
        self.timeline_view.set_presence(presence)
        self.timeline_view.set_duplicates(duplicates)
        self.timeline_view.set_current_frame(self.current_frame_index)

    def on_timeline_frame_selected(self, frame_idx: int):
        """Jump to a frame when clicking the timeline."""
        self.save_current_annotations()
        self.current_frame_index = frame_idx
        self.frame_slider.blockSignals(True)
        self.frame_slider.setValue(self.current_frame_index)
        self.frame_slider.blockSignals(False)
        self.display_frame()
        self.timeline_view.set_current_frame(self.current_frame_index)
        print(f"Timeline jump to frame: {frame_idx}")

    def on_timeline_objects_selected(self, obj_ids):
        """Sync selection from timeline bar clicks."""
        new_selected = set(obj_ids)
        if new_selected != self.selected_object_ids:
            self.selected_object_ids = new_selected
            self.sync_current_frame_objects_list()
            self.merge_button.setEnabled(len(self.selected_object_ids) > 1)
            has_selection = len(self.selected_object_ids) > 0
            self.delete_object_button.setEnabled(has_selection)
            self.delete_id_global_button.setEnabled(has_selection)
            self.timeline_view.set_selected_object_ids(self.selected_object_ids)
            self.display_frame()

    def on_timeline_segments_selected(self, segments):
        """Handle selection ranges drawn on the timeline."""
        self.timeline_selected_segments = segments
        self.delete_segments_button.setEnabled(len(segments) > 0)
        if segments:
            obj_ids = {seg[0] for seg in segments}
            self.selected_object_ids = obj_ids
            self.timeline_view.set_selected_object_ids(obj_ids)
            self.sync_current_frame_objects_list()
            has_selection = len(self.selected_object_ids) > 0
            self.merge_button.setEnabled(len(self.selected_object_ids) > 1)
            self.delete_object_button.setEnabled(has_selection)
            self.delete_id_global_button.setEnabled(has_selection)
                 
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
        self.refresh_timeline()

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
        self.mark_clean()

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
            if self.timeline_selected_segments:
                self.delete_selected_ranges()
            else:
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
            # Update button states
            self.merge_button.setEnabled(len(self.selected_object_ids) > 1)
            self.delete_object_button.setEnabled(len(self.selected_object_ids) > 0)
            self.delete_id_global_button.setEnabled(len(self.selected_object_ids) > 0)
            self.timeline_view.set_selected_object_ids(self.selected_object_ids)
            # Refresh display to show highlights
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
        has_selection = len(self.selected_object_ids) > 0
        self.merge_button.setEnabled(len(self.selected_object_ids) > 1)
        self.delete_object_button.setEnabled(has_selection)
        self.delete_id_global_button.setEnabled(has_selection)
    
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
        self.selected_object_ids.clear()
        self.merge_button.setEnabled(False)
        self.delete_object_button.setEnabled(False)
        self.delete_id_global_button.setEnabled(False)
        self.timeline_view.set_selected_object_ids(self.selected_object_ids)
        
        # Refresh current frame display
        self.display_frame()
        self.refresh_timeline()
        self.mark_dirty()
        
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
        self.refresh_timeline()
        self.mark_dirty()
        
    def delete_selected_ids_globally(self):
        """Delete all instances of selected object IDs across every frame."""
        if len(self.selected_object_ids) == 0:
            return
        ids_to_delete = set(self.selected_object_ids)
        total_deleted = 0

        for frame_idx in range(self.total_frames):
            before = len(self.annotations[frame_idx])
            self.annotations[frame_idx] = [
                (obj_id, class_id, x_min, y_min, x_max, y_max)
                for obj_id, class_id, x_min, y_min, x_max, y_max in self.annotations[frame_idx]
                if obj_id not in ids_to_delete
            ]
            total_deleted += before - len(self.annotations[frame_idx])

        self.all_object_ids.difference_update(ids_to_delete)

        self.selected_object_ids.clear()
        self.merge_button.setEnabled(False)
        self.delete_object_button.setEnabled(False)
        self.delete_id_global_button.setEnabled(False)
        self.timeline_view.set_selected_object_ids(self.selected_object_ids)

        self.refresh_timeline()
        self.display_frame()
        print(f"Deleted {total_deleted} boxes with object ID(s) {ids_to_delete} across all frames")
        self.mark_dirty()

    def delete_selected_ranges(self):
        """Delete boxes only within selected timeline ranges for specific object IDs."""
        if not self.timeline_selected_segments:
            return
        total_deleted = 0
        for obj_id, start_f, end_f in self.timeline_selected_segments:
            for frame_idx in range(max(0, start_f), min(self.total_frames - 1, end_f) + 1):
                before = len(self.annotations[frame_idx])
                self.annotations[frame_idx] = [
                    (oid, cls, x1, y1, x2, y2)
                    for (oid, cls, x1, y1, x2, y2) in self.annotations[frame_idx]
                    if oid != obj_id
                ]
                total_deleted += before - len(self.annotations[frame_idx])
        print(f"Deleted {total_deleted} box(es) in selected ranges: {self.timeline_selected_segments}")
        self.timeline_selected_segments = []
        self.timeline_view.clear_selected_segments()
        self.delete_segments_button.setEnabled(False)
        self.display_frame()
        self.refresh_timeline()
        self.mark_dirty()

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
