import sys
import json
import math
import os
import random
import shutil
import subprocess
import numpy as np
import librosa
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
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QSpinBox,
    QCheckBox,
    QMessageBox,
)
from PyQt5.QtGui import QImage, QPixmap, QColor, QPen, QBrush, QFont, QPolygonF, QPalette, QFontMetrics
from PyQt5.QtCore import Qt, QRectF, pyqtSignal, QPointF, QPoint, QUrl, QTimer
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget
import cv2
import argparse
from .bounding_box import BoundingBoxItem, generate_color_for_id
from .annotator_view import AnnotationView 
from .inference import YOLOInferenceRunner 
from .box_downloader import BoxNavigator


PROJECT_ROOT = os.path.abspath(os.getcwd())
ANNOTATIONS_ROOT = os.path.join(PROJECT_ROOT, "annotations")
TEST_ROOT = os.path.join(PROJECT_ROOT, "test")
SYSTEM_FILES_DIR = os.path.join(PROJECT_ROOT, "system_files")
OUTPUT_ROOT = TEST_ROOT
DATASET_DIR = os.path.join(OUTPUT_ROOT, "data")
VIDEO_DIR = os.path.join(OUTPUT_ROOT, "videos")

def configure_paths(base_dir: str) -> None:
    global PROJECT_ROOT, ANNOTATIONS_ROOT, TEST_ROOT, SYSTEM_FILES_DIR
    PROJECT_ROOT = os.path.abspath(base_dir)
    ANNOTATIONS_ROOT = os.path.join(PROJECT_ROOT, "annotations")
    TEST_ROOT = os.path.join(PROJECT_ROOT, "test")
    SYSTEM_FILES_DIR = os.path.join(PROJECT_ROOT, "system_files")

class NoWheelScrollArea(QScrollArea):
    """Scroll area that ignores wheel events to keep scrolling exclusive to timeline."""
    def wheelEvent(self, event):
        event.ignore()

# Place helper functions BEFORE the class that uses them
def load_yolo_to_pixel(yolo_dir, frame_index, frame_width, frame_height):
    """Reads YOLO .txt for one frame and converts to pixel coordinates.
    Supports old format (5 values), object_id format (6 values),
    and object_id + confidence format (7 values).
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
            if len(parts) == 7:
                # New format: object_id class_id conf x_c y_c w h
                object_id = int(parts[0])
                class_id = int(parts[1])
                confidence = float(parts[2])
                xc, yc, w_norm, h_norm = [float(p) for p in parts[3:]]
            elif len(parts) == 6:
                # New format: object_id class_id x_c y_c w h
                object_id = int(parts[0])
                class_id = int(parts[1])
                confidence = None
                xc, yc, w_norm, h_norm = [float(p) for p in parts[2:]]
            elif len(parts) == 5:
                # Old format: class_id x_c y_c w h (backward compatibility)
                # Assign sequential object_id for backward compatibility
                object_id = sequential_id
                sequential_id += 1
                class_id = int(parts[0])
                confidence = None
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
                
            # Store as (object_id, class_id, x_min, y_min, x_max, y_max, confidence)
            annotations.append((object_id, class_id, x_min, y_min, x_max, y_max, confidence))
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
        raise ValueError("Either --video-path or --box-video is required.")

    navigator = BoxNavigator(
        base_dir=OUTPUT_ROOT,
        system_files_dir=SYSTEM_FILES_DIR,
        download_dir=VIDEO_DIR,
    )
    downloaded_path = navigator.download_vid(box_video_name)
    if not downloaded_path:
        raise FileNotFoundError(f"Could not download '{box_video_name}' from Box.")
    return downloaded_path


def confirm_cross_split(video_stem: str, video_name: str, other_root: str) -> bool:
    """Warn if the video exists in the other split; return True to proceed."""
    other_data = os.path.join(other_root, "data", video_stem)
    other_video = os.path.join(other_root, "videos", video_name)
    if not (os.path.exists(other_data) or os.path.exists(other_video)):
        return True
    details = []
    if os.path.exists(other_data):
        details.append(f"- data: {other_data}")
    if os.path.exists(other_video):
        details.append(f"- video: {other_video}")
    message = "\n".join(
        [
            f"[warn] '{video_name}' already exists in the other split.",
            *details,
            "Annotations/test should not share videos.",
        ]
    )
    print(message)
    response = input("Proceed anyway? [y/N]: ").strip().lower()
    return response in ("y", "yes")

def create_random_snippet(video_path: str, duration_seconds: float = 5.0) -> str:
    """
    Create a random ~5s snippet with audio using stream copy (no re-encode).
    Falls back to the original video if clipping fails or ffmpeg is unavailable.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Could not open video at {video_path}, using original.")
        return video_path

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    frames_needed = max(1, int(duration_seconds * fps))
    if total_frames <= frames_needed or frame_width == 0 or frame_height == 0:
        print(f"Video shorter than {duration_seconds}s; using full video.")
        return video_path

    start_frame = random.randint(0, total_frames - frames_needed)
    actual_duration = min(duration_seconds, (total_frames - start_frame) / fps)

    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        print("ffmpeg not found; using original video.")
        return video_path

    snippet_dir = os.path.join(OUTPUT_ROOT, "video_snippets")
    os.makedirs(snippet_dir, exist_ok=True)
    snippet_name = f"{os.path.splitext(os.path.basename(video_path))[0]}_snippet_{start_frame}.mp4"
    snippet_path = os.path.join(snippet_dir, snippet_name)

    # Use stream copy to avoid re-encoding (fast, preserves quality and audio)
    cmd = [
        ffmpeg_bin,
        "-y",
        "-ss",
        f"{start_frame / fps:.3f}",
        "-t",
        f"{actual_duration:.3f}",
        "-i",
        video_path,
        "-c",
        "copy",
        "-movflags",
        "+faststart",
        snippet_path,
    ]

    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        print("ffmpeg failed to create snippet; using original video.")
        return video_path

    print(
        f"Using random 5s snippet with audio: start_frame={start_frame}, "
        f"duration={actual_duration:.2f}s saved to {snippet_path}"
    )
    return snippet_path

SNIPPET_INFO_FILENAME = "snippet_info.json"

def _snippet_manifest_path(dataset_stem: str) -> str:
    return os.path.join(DATASET_DIR, dataset_stem, SNIPPET_INFO_FILENAME)

def load_snippet_from_manifest(dataset_stem: str) -> str:
    """Return snippet path from manifest if it exists and is readable."""
    manifest_path = _snippet_manifest_path(dataset_stem)
    if not os.path.exists(manifest_path):
        return None
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        candidate = data.get("snippet_path") or data.get("video_path")
        if candidate and os.path.exists(candidate):
            return candidate
    except Exception as exc:
        print(f"Failed to read snippet manifest at {manifest_path}: {exc}")
    return None

def save_snippet_manifest(dataset_stem: str, snippet_path: str, source_video_path: str = None):
    """Persist the snippet used for annotation to reload cached datasets correctly."""
    manifest_path = _snippet_manifest_path(dataset_stem)
    try:
        os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
        payload = {
            "snippet_path": snippet_path,
            "source_video_path": source_video_path or snippet_path,
        }
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    except Exception as exc:
        print(f"Failed to save snippet manifest: {exc}")

def find_existing_snippet(dataset_stem: str) -> str:
    """Best-effort lookup for an existing snippet matching the dataset stem."""
    snippet_dir = os.path.join(OUTPUT_ROOT, "video_snippets")
    if not os.path.isdir(snippet_dir):
        return None
    candidates = [
        os.path.join(snippet_dir, f)
        for f in os.listdir(snippet_dir)
        if f.startswith(f"{dataset_stem}_snippet_") and f.endswith(".mp4")
    ]
    if not candidates:
        return None
    # Pick most recently modified to favor the latest annotation run
    return max(candidates, key=os.path.getmtime)

def probe_readable_frame_count(video_path: str, reported_frames: int) -> int:
    """
    Some containers over-report frame counts; probe the tail to clamp to readable frames.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return reported_frames

    last_valid = None
    probe = max(reported_frames - 1, 0)
    attempts = 0
    while probe >= 0 and attempts < 30:
        cap.set(cv2.CAP_PROP_POS_FRAMES, probe)
        ret, _ = cap.read()
        if ret:
            last_valid = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            break
        probe -= 1
        attempts += 1

    if last_valid is None:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        count = 0
        while count < reported_frames:
            ret, _ = cap.read()
            if not ret:
                break
            count += 1
        cap.release()
        return count if count > 0 else reported_frames

    cap.release()
    return last_valid

def jump_label_color(label: str) -> QColor:
    """Consistent colors for jump window labels."""
    if label == "snap":
        return QColor(200, 90, 90)
    if label == "grunt":
        return QColor(90, 170, 220)
    return QColor(140, 140, 140)

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
        self.bar_height = 6
        self.row_spacing = 4
        self.setMinimumHeight(110)
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
        # Any selection rectangle gets deleted when the scene is cleared; drop the reference
        # to avoid calling methods on a freed Qt item after a redraw.
        self.selection_rect_item = None
        self.selecting = False
        self.selection_start = None
        self.bar_geometries = []
        fm = QFontMetrics(QFont())
        max_id = max(self.presence.keys()) if self.presence else 0
        label_width = fm.boundingRect(f"ID {max_id}").width() + 12
        # Margin wide enough for labels; minimum keeps layout stable
        self.left_margin = max(70, label_width)
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
            # Draw object ID label at start of row
            base_font = QFont()
            label_font = QFont(base_font)
            r, g, b = generate_color_for_id(obj_id)
            color = QColor(r, g, b)
            is_selected = obj_id in self.selected_object_ids
            if is_selected:
                label_font.setBold(True)
                label_color = color.lighter(130)
            else:
                label_color = color
            label_item = self.scene_obj.addText(f"ID {obj_id}", label_font)
            label_item.setDefaultTextColor(label_color)
            label_item.setPos(4, y - 2)
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
                if self.selection_rect_item.scene() == self.scene_obj:
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
                if self.selection_rect_item.scene() == self.scene_obj:
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
            if self.selection_rect_item and self.selection_rect_item.scene() == self.scene_obj:
                sel_rect = self.selection_rect_item.rect()
                self.scene_obj.removeItem(self.selection_rect_item)
                self.selection_rect_item = None
                self._apply_selection_rect(sel_rect)
            else:
                # Selection rect was cleared during a redraw; just reset state.
                self.selection_rect_item = None
                self.selection_start = None
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

class JumpTimelineView(QGraphicsView):
    """Single-lane timeline for jump windows."""
    frame_selected = pyqtSignal(int)
    window_selected = pyqtSignal(int)
    delete_requested = pyqtSignal()

    def __init__(self, total_frames: int, parent=None):
        self.scene_obj = QGraphicsScene(parent)
        super().__init__(self.scene_obj, parent)
        self.total_frames = max(total_frames, 1)
        fm = QFontMetrics(QFont())
        self.left_margin = max(60, fm.boundingRect("Jump").width() + 12)
        self.base_width = 600
        self.timeline_width = self.base_width
        self.bar_height = 8
        self.row_spacing = 3
        self.setMinimumHeight(80)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setFrameStyle(QFrame.NoFrame)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.bg_color = self.palette().color(QPalette.Base)
        self.zoom_factor = 1.0
        self.zoom_min = 1.0
        self.zoom_max = 5.0
        self.current_frame = 0
        self.windows = []
        self.bar_geometries = []
        self.selected_index = None
        self.dragging_marker = False
        self.provisional_start = None
        self._redraw()

    def set_total_frames(self, total_frames: int):
        self.total_frames = max(total_frames, 1)
        self._redraw()

    def set_windows(self, windows: list):
        self.windows = list(windows)
        self._redraw()

    def set_current_frame(self, frame_idx: int):
        self.current_frame = max(0, min(frame_idx, self.total_frames - 1))
        self._redraw()

    def set_selected_index(self, idx: int):
        self.selected_index = idx
        self._redraw()

    def set_provisional_start(self, frame_idx: int):
        self.provisional_start = max(0, min(frame_idx, self.total_frames - 1))
        self._redraw()

    def clear_provisional_start(self):
        self.provisional_start = None
        self._redraw()

    def resizeEvent(self, event):
        self.base_width = max(200, self.viewport().width() - self.left_margin - 20)
        self.timeline_width = max(200, self.base_width * self.zoom_factor)
        self._redraw()
        super().resizeEvent(event)

    def zoom_in(self):
        if self.zoom_factor < self.zoom_max:
            self.zoom_factor *= 1.2
            self._redraw()

    def zoom_out(self):
        if self.zoom_factor > self.zoom_min:
            self.zoom_factor /= 1.2
            self._redraw()

    def _frame_from_pos(self, pos_x: float, scale: float):
        frame = int((pos_x - self.left_margin) / scale)
        return max(0, min(frame, self.total_frames - 1))

    def _redraw(self):
        self.scene_obj.clear()
        self.bar_geometries = []
        fm = QFontMetrics(QFont())
        self.left_margin = max(60, fm.boundingRect("Jump").width() + 12)
        self.base_width = max(200, self.viewport().width() - self.left_margin - 20)
        self.timeline_width = max(200, self.base_width * self.zoom_factor)
        scale = self.timeline_width / float(max(self.total_frames - 1, 1))
        lane_y = 8

        # Label column
        label_rect = QRectF(0, 0, self.left_margin - 6, lane_y + self.bar_height + 12)
        self.scene_obj.addRect(label_rect, QPen(Qt.NoPen), QBrush(self.bg_color))
        label_item = self.scene_obj.addText("Jump", QFont())
        label_item.setDefaultTextColor(self.palette().color(QPalette.Text))
        label_item.setPos(6, lane_y)

        # Background bar
        bar_rect = QRectF(self.left_margin, lane_y, self.timeline_width, self.bar_height)
        self.scene_obj.addRect(bar_rect, QPen(QColor(200, 200, 200)), QBrush(QColor(235, 235, 235)))

        # Jump windows
        for idx, win in enumerate(self.windows):
            start_x = self.left_margin + win["start_frame"] * scale
            end_x = self.left_margin + win["end_frame"] * scale
            width = max(3, end_x - start_x)
            color = jump_label_color(win.get("label"))
            pen = QPen(color.darker(130), 2 if self.selected_index == idx else 1)
            rect_item = self.scene_obj.addRect(start_x, lane_y, width, self.bar_height, pen, QBrush(color))
            rect_item.setOpacity(0.75)
            self.bar_geometries.append((idx, QRectF(start_x, lane_y, width, self.bar_height)))

        # Provisional start marker
        if self.provisional_start is not None:
            start_x = self.left_margin + self.provisional_start * scale
            start_pen = QPen(QColor(140, 140, 140), 2, Qt.SolidLine)
            self.scene_obj.addLine(start_x, lane_y - 4, start_x, lane_y + self.bar_height + 4, start_pen)
            start_rect = QRectF(start_x - 1.5, lane_y, 3, self.bar_height)
            self.scene_obj.addRect(start_rect, QPen(Qt.NoPen), QBrush(QColor(170, 170, 170, 150)))

        # Current frame marker
        marker_x = self.left_margin + self.current_frame * scale
        marker_pen = QPen(QColor(50, 120, 220), 2)
        self.scene_obj.addLine(marker_x, lane_y - 6, marker_x, lane_y + self.bar_height + 6, marker_pen)

        # Set scene rect for scrolling
        full_height = lane_y + self.bar_height + 16
        self.scene_obj.setSceneRect(0, 0, self.left_margin + self.timeline_width + 10, full_height)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            pos = self.mapToScene(event.pos())
            scale = self.timeline_width / float(max(self.total_frames - 1, 1))
            frame = self._frame_from_pos(pos.x(), scale)
            self.set_current_frame(frame)
            self.frame_selected.emit(frame)
            self.dragging_marker = True

            # Hit test bars
            clicked_idx = None
            for idx, rect in self.bar_geometries:
                if rect.contains(pos):
                    clicked_idx = idx
                    break
            if clicked_idx is not None:
                self.selected_index = clicked_idx
                self.window_selected.emit(clicked_idx)
                self._redraw()
            event.accept()
            return
        if event.button() == Qt.RightButton:
            pos = self.mapToScene(event.pos())
            scale = self.timeline_width / float(max(self.total_frames - 1, 1))
            frame = self._frame_from_pos(pos.x(), scale)
            self.set_current_frame(frame)
            self.frame_selected.emit(frame)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.dragging_marker:
            pos = self.mapToScene(event.pos())
            scale = self.timeline_width / float(max(self.total_frames - 1, 1))
            frame = self._frame_from_pos(pos.x(), scale)
            if frame != self.current_frame:
                self.set_current_frame(frame)
                self.frame_selected.emit(frame)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.dragging_marker and event.button() == Qt.LeftButton:
            self.dragging_marker = False
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event):
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
            self.delete_requested.emit()
            event.accept()
            return
        super().keyPressEvent(event)


class MainWindow(QMainWindow):
    def __init__(self, video_path, yolo_weights, save_video=False, save_clips=False, run_new=False, dataset_stem=None, source_video_path=None, write_snippet_manifest=False):
        super().__init__()
        self.setWindowTitle("YOLO Video Annotator")
        self.setFocusPolicy(Qt.StrongFocus)
        self.save_video = save_video
        self.save_clips = save_clips
        self.run_new = run_new
        self.write_snippet_manifest = write_snippet_manifest

        # --- 1. Video and YOLO Setup ---
        self.video_path = video_path
        self.source_video_path = source_video_path or video_path
        self.dataset_stem = dataset_stem or os.path.splitext(os.path.basename(self.source_video_path))[0]
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            print(f'Error loading video file: {video_path}')
            sys.exit(1)
            
        # Decide whether to reuse cached labels or run YOLO
        self.using_cached_labels = False
        self.yolo_dir = None
        self.yolo_predictions = None
        cached_labels_dir = os.path.join(DATASET_DIR, self.dataset_stem, "labels")
        if (not self.run_new) and os.path.isdir(cached_labels_dir) and any(f.endswith(".txt") for f in os.listdir(cached_labels_dir)):
            print(f"Using cached labels from {cached_labels_dir}")
            self.yolo_dir = cached_labels_dir
            self.using_cached_labels = True
        else:
            yolo_runner = YOLOInferenceRunner(yolo_weights)
            self.yolo_predictions = yolo_runner.run_inference_in_memory(self.video_path)
        
        # Get frame properties and clamp total_frames to actually readable frames
        reported_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        actual_frames = probe_readable_frame_count(self.video_path, reported_frames)
        if actual_frames != reported_frames:
            print(f"Adjusted frame count from {reported_frames} to {actual_frames} based on readable frames.")
        self.total_frames = max(actual_frames, 1)
        self.current_frame_index = 0
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 30.0)
        self.video_stem = os.path.splitext(os.path.basename(self.video_path))[0]
        self.jump_audio_root = os.path.join(OUTPUT_ROOT, "jump_audio_data", self.dataset_stem)

        # Load annotations from disk
        self.annotations = [[] for i in range(self.total_frames)]
        self.next_object_id = 0  # For assigning IDs to manually added boxes
        self.all_object_ids = set()  # Track all unique object IDs across all frames
        for i in range(self.total_frames):
            if self.using_cached_labels:
                frame_annotations = load_yolo_to_pixel(self.yolo_dir, i, self.frame_width, self.frame_height)
            else:
                if self.yolo_predictions and i < len(self.yolo_predictions):
                    frame_annotations = self.yolo_predictions[i]
                else:
                    frame_annotations = []
            self.annotations[i] = frame_annotations
            # Track the highest object_id to assign new ones and collect all IDs
            for obj_id, _, _, _, _, _, _ in frame_annotations:
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
        self.scene.selectionChanged.connect(self.on_scene_selection_changed)
        
        self.pixmap_item = QGraphicsPixmapItem()
        self.scene.addItem(self.pixmap_item)

        # Object list display showing color, object ID, and class
        header_row = QHBoxLayout()
        objects_label = QLabel("Objects in Current Frame:")
        header_row.addWidget(objects_label)
        header_row.addStretch(1)
        self.btn_save = QPushButton('Save dataset')
        self.btn_save.setFixedHeight(28)
        header_row.addWidget(self.btn_save)
        self.layout.addLayout(header_row)
        self.objects_list = QListWidget()
        self.objects_list.setMaximumHeight(60)
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
        self.merge_all_button = QPushButton("Merge All")
        self.merge_all_button.setEnabled(len(self.all_object_ids) > 1)
        self.merge_all_button.clicked.connect(self.merge_all_objects)
        button_layout.addWidget(self.merge_all_button)
        
        self.delete_object_button = QPushButton("Delete from Frame")
        self.delete_object_button.setEnabled(False)
        self.delete_object_button.clicked.connect(self.delete_selected_objects_from_frame)
        button_layout.addWidget(self.delete_object_button)
        self.delete_id_global_button = QPushButton("Delete Object From Video")
        self.delete_id_global_button.setEnabled(False)
        self.delete_id_global_button.clicked.connect(self.delete_selected_ids_globally)
        button_layout.addWidget(self.delete_id_global_button)
        self.propagate_button = QPushButton("Propagate Box...")
        self.propagate_button.setFixedHeight(28)
        self.propagate_button.setFocusPolicy(Qt.NoFocus)
        self.propagate_button.setEnabled(False)
        self.propagate_button.clicked.connect(self.open_propagate_dialog)
        button_layout.addWidget(self.propagate_button)
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
        self.btn_timeline_zoom_out.setFocusPolicy(Qt.NoFocus)
        self.btn_timeline_zoom_in = QPushButton("Zoom +")
        self.btn_timeline_zoom_in.setMaximumWidth(70)
        self.btn_timeline_zoom_in.setFocusPolicy(Qt.NoFocus)
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

        # Jump windows timeline
        jump_header = QHBoxLayout()
        jump_label = QLabel("Jump Windows Timeline:")
        jump_label.setStyleSheet("font-size: 9pt; font-weight: bold;")
        jump_header.addWidget(jump_label)
        jump_header.addStretch(1)
        self.btn_jump_start = QPushButton("Start Jump (J)")
        self.btn_jump_end = QPushButton("End Jump (K)")
        self.btn_jump_end.setEnabled(False)
        self.btn_delete_jump = QPushButton("Delete Selected")
        # Legend for jump sounds
        legend_font = self.btn_jump_start.font()
        legend_label = QLabel("Key:")
        legend_label.setFont(legend_font)
        jump_header.addWidget(legend_label)
        legend_style = "padding: 2px 6px; border: 1px solid #444; border-radius: 3px;"
        legend_snap = QLabel("Snap")
        legend_snap.setFont(legend_font)
        legend_snap.setStyleSheet(f"{legend_style} background-color: rgb(200, 90, 90); color: black;")
        legend_grunt = QLabel("Grunt")
        legend_grunt.setFont(legend_font)
        legend_grunt.setStyleSheet(f"{legend_style} background-color: rgb(90, 170, 220); color: black;")
        legend_none = QLabel("None")
        legend_none.setFont(legend_font)
        legend_none.setStyleSheet(f"{legend_style} background-color: rgb(140, 140, 140); color: black;")
        jump_header.addWidget(legend_snap)
        jump_header.addWidget(legend_grunt)
        jump_header.addWidget(legend_none)
        jump_header.addSpacing(12)
        jump_header.addWidget(QLabel("Sound:"))
        self.jump_sound_combo = QComboBox()
        self.jump_sound_combo.addItems(["none", "snap", "grunt"])
        self.jump_sound_combo.setEnabled(False)
        jump_header.addWidget(self.jump_sound_combo)
        for btn in (self.btn_jump_start, self.btn_jump_end, self.btn_delete_jump):
            btn.setFixedHeight(28)
            btn.setFocusPolicy(Qt.NoFocus)
        jump_header.addWidget(self.btn_jump_start)
        jump_header.addWidget(self.btn_jump_end)
        jump_header.addWidget(self.btn_delete_jump)
        bottom_layout.addLayout(jump_header)

        self.jump_timeline_view = JumpTimelineView(self.total_frames)
        self.jump_timeline_view.setMaximumHeight(100)
        bottom_layout.addWidget(self.jump_timeline_view)
        # Make bottom widget scrollable
        scroll_area = NoWheelScrollArea()
        scroll_area.setWidget(bottom_widget)
        scroll_area.setWidgetResizable(True)
        scroll_area.setMaximumHeight(260)
        scroll_area.setMinimumHeight(180)
        
        # Add bottom section to splitter
        splitter.addWidget(scroll_area)
        
        # Set splitter sizes (give more space to timeline than before)
        splitter.setSizes([560, 400])
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
        self.btn_play_pause = QPushButton("Play")
        self.btn_play_pause.setFixedHeight(28)
        self.btn_next = QPushButton("Next Frame >>")
        self.btn_next.setFixedHeight(28)
        button_layout.addWidget(self.btn_prev)
        button_layout.addWidget(self.btn_play_pause)
        button_layout.addWidget(self.btn_next)
        button_widget = QWidget()
        button_widget.setLayout(button_layout)
        self.layout.addWidget(button_widget)
        self.data_dirty = True  # Start dirty until first explicit save
        self._update_save_button_label()
        
        # --- 3. Connections and Initial Display ---
        self.btn_next.clicked.connect(self.go_to_next_frame)
        self.btn_prev.clicked.connect(self.go_to_previous_frame)
        self.btn_save.clicked.connect(self.save_dataset)
        # Ignore the clicked(bool) payload so play/pause toggles correctly
        self.btn_play_pause.clicked.connect(lambda: self.toggle_playback())

        # Jump windows state
        self.jump_windows = []
        self.jump_start_frame = None
        self.jump_end_frame = None
        self.selected_jump_index = None
        self.last_propagate_count = 10
        self.propagate_stop_on_existing = True
        self.propagate_overwrite_existing = False
        self.playing = False
        self.play_timer = QTimer()
        self.play_timer.setTimerType(Qt.PreciseTimer)
        self.play_timer.timeout.connect(self.advance_playback)
        # Attach a hidden video surface to avoid QtMultimedia crashes on macOS when no surface exists.
        self._audio_dummy_surface = QVideoWidget()
        self._audio_dummy_surface.hide()
        self.audio_player = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        self.audio_player.setVideoOutput(self._audio_dummy_surface)
        self.audio_player.setMedia(QMediaContent(QUrl.fromLocalFile(self.video_path)))
        self.audio_player.setVolume(80)

        # Jump windows connections
        self.btn_jump_start.clicked.connect(self.mark_jump_start)
        self.btn_jump_end.clicked.connect(self.mark_jump_end)
        self.btn_delete_jump.clicked.connect(self.delete_selected_jump_window)
        self.jump_timeline_view.frame_selected.connect(self.on_jump_timeline_frame_selected)
        self.jump_timeline_view.window_selected.connect(self.on_jump_window_selected)
        self.jump_timeline_view.delete_requested.connect(self.delete_selected_jump_window)
        self.jump_sound_combo.currentIndexChanged.connect(self.set_selected_jump_sound)
        if self.using_cached_labels:
            self.load_jump_manifest()

        self.refresh_timeline()
        self.refresh_jump_timeline()
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
        # Stop playback cleanly on save completion
        if self.playing:
            self.toggle_playback(force_state=False)

    def update_merge_all_button_state(self):
        """Enable merge-all when multiple object IDs exist."""
        if hasattr(self, "merge_all_button"):
            self.merge_all_button.setEnabled(len(self.all_object_ids) > 1)

    def _scene_has_selected_box(self) -> bool:
        """Return True if any scene item selection is a bounding box."""
        return any(isinstance(item, BoundingBoxItem) for item in self.scene.selectedItems())

    def _update_propagate_button_state(self):
        if hasattr(self, "propagate_button"):
            self.propagate_button.setEnabled(self._scene_has_selected_box())

    def _determine_draw_object_id(self):
        """Return the object_id to use when drawing a new box."""
        if self.timeline_selected_segments:
            return self.timeline_selected_segments[0][0]
        if self.selected_object_ids:
            return sorted(self.selected_object_ids)[0]
        return None

    def save_current_annotations(self):
        """Persist current frame's boxes from the scene into the annotations list."""
        previous = list(self.annotations[self.current_frame_index])
        new_entries = []

        for item in self.scene.items():
            # Check if the item is one of our custom bounding boxes
            if isinstance(item, BoundingBoxItem):
                # Use scene coordinates so moves/resizes are captured
                rect = item.mapRectToScene(item.rect())

                box_data = (
                    item.object_id,
                    item.class_id,
                    int(rect.left()),
                    int(rect.top()),
                    int(rect.right()),
                    int(rect.bottom()),
                    item.confidence,
                )
                # Store as (object_id, class_id, x_min, y_min, x_max, y_max, confidence)
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
        box_item = BoundingBoxItem(rect, class_id=0, object_id=target_obj_id, on_change=self.on_box_changed) 
        self.scene.addItem(box_item)
        
        # 3. Add the pixel coordinates to the current frame's data structure
        new_box_data = (
            target_obj_id,
            box_item.class_id,
            int(rect.left()),
            int(rect.top()),
            int(rect.right()),
            int(rect.bottom()),
            None,
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
        self._update_propagate_button_state()
        
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
        
        if not ret or frame is None:
            print(f"Error reading frame at index {self.current_frame_index}.")
            return

        # Make a contiguous RGB copy before handing off to Qt to avoid dangling buffers.
        rgb_image = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        rgb_bytes = rgb_image.tobytes()  # force an owned copy in Python memory
        qt_image = QImage(rgb_bytes, w, h, bytes_per_line, QImage.Format_RGB888).copy()
        pixmap = QPixmap.fromImage(qt_image)
        # Keep a reference so the backing buffer is not GC'd mid-paint (belt-and-suspenders).
        self._last_frame_bytes = rgb_bytes

        # Update the QGraphicsPixmapItem
        self.pixmap_item.setPixmap(pixmap)
        # Allow zoom to persist; only auto-fit if user has not zoomed
        self.view.setSceneRect(self.pixmap_item.boundingRect())
        if not self.view.user_zoomed:
            self.view.fit_to_pixmap(self.pixmap_item)
        
        # --- Bounding Box Drawing Logic ---
        # Clear ALL old bounding boxes from the scene
        items_to_remove = [item for item in self.scene.items() if isinstance(item, BoundingBoxItem)]
        for item in items_to_remove:
            # Defensive: only remove from the scene that actually owns the item
            item_scene = item.scene()
            if item_scene is not None:
                item_scene.removeItem(item)

        # Draw the bounding boxes for the current frame
        boxes = self.annotations[self.current_frame_index]
        for obj_id, class_id, x_min, y_min, x_max, y_max, confidence in boxes:
            rect_qt = QRectF(x_min, y_min, x_max - x_min, y_max - y_min)
            # Check if this object ID is selected for highlighting
            is_highlighted = obj_id in self.selected_object_ids
            box_item = BoundingBoxItem(
                rect_qt,
                class_id=class_id,
                object_id=obj_id,
                confidence=confidence,
                is_highlighted=is_highlighted,
                on_change=self.on_box_changed,
            )
            self.scene.addItem(box_item)
        self._update_propagate_button_state()
        self.update_box_count()
        self.timeline_view.set_current_frame(self.current_frame_index)
        self.update_merge_all_button_state()

    def on_box_changed(self, _item):
        """Autosave when a box is moved or resized."""
        self.save_current_annotations()
        self.refresh_timeline()

    def refresh_timeline(self):
        """Rebuild object timeline presence map and refresh the view."""
        presence = {}
        duplicates = {}
        for frame_idx, frame_annotations in enumerate(self.annotations):
            counts = {}
            for obj_id, _, _, _, _, _, _ in frame_annotations:
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
        self.update_merge_all_button_state()

    def refresh_jump_timeline(self):
        selected_window = None
        if self.selected_jump_index is not None and 0 <= self.selected_jump_index < len(self.jump_windows):
            selected_window = (
                self.jump_windows[self.selected_jump_index]["start_frame"],
                self.jump_windows[self.selected_jump_index]["end_frame"],
            )

        self.jump_windows = sorted(self.jump_windows, key=lambda w: w["start_frame"])
        self.jump_timeline_view.set_total_frames(self.total_frames)
        self.jump_timeline_view.set_windows(self.jump_windows)
        self.jump_timeline_view.set_current_frame(self.current_frame_index)
        if selected_window:
            self.selected_jump_index = next(
                (i for i, w in enumerate(self.jump_windows) if w["start_frame"] == selected_window[0] and w["end_frame"] == selected_window[1]),
                None,
            )
        if self.selected_jump_index is not None:
            self.jump_timeline_view.set_selected_index(self.selected_jump_index)
            if 0 <= self.selected_jump_index < len(self.jump_windows):
                current_label = self.jump_windows[self.selected_jump_index]["label"]
                idx = self.jump_sound_combo.findText(current_label)
                if idx != -1:
                    self.jump_sound_combo.blockSignals(True)
                    self.jump_sound_combo.setCurrentIndex(idx)
                    self.jump_sound_combo.blockSignals(False)
                self.jump_sound_combo.setEnabled(True)
        else:
            self.jump_sound_combo.blockSignals(True)
            self.jump_sound_combo.setCurrentIndex(self.jump_sound_combo.findText("none"))
            self.jump_sound_combo.blockSignals(False)
            self.jump_sound_combo.setEnabled(False)

    # ---------- Jump window helpers ----------
    def mark_jump_start(self):
        self.jump_start_frame = self.current_frame_index
        self.btn_jump_end.setEnabled(True)
        print(f"Jump start set to frame {self.jump_start_frame}")
        self.jump_timeline_view.setFocus()
        # Show provisional start marker until end is chosen
        self.jump_timeline_view.set_provisional_start(self.jump_start_frame)

    def mark_jump_end(self):
        if self.jump_start_frame is None:
            print("Select a start frame first.")
            return
        if self.current_frame_index <= self.jump_start_frame:
            print("End frame must come after start frame.")
            return
        self.jump_end_frame = self.current_frame_index
        self.add_jump_window(self.jump_start_frame, self.jump_end_frame, "none")
        self.jump_start_frame = None
        self.jump_end_frame = None
        self.btn_jump_end.setEnabled(False)
        print(f"Jump end set to frame {self.current_frame_index} with label none")
        self.jump_timeline_view.setFocus()
        self.jump_timeline_view.clear_provisional_start()

    def _jump_manifest_path(self):
        return os.path.join(self.jump_audio_root, "manifest.json")

    def _jump_clip_path(self, start_frame, end_frame, label):
        fname = f"{start_frame:05d}_{end_frame:05d}_{label}.wav"
        return os.path.join(self.jump_audio_root, "clips", fname)

    def _jump_spec_paths(self, start_frame, end_frame, label):
        base = f"{start_frame:05d}_{end_frame:05d}_{label}"
        spec_dir = os.path.join(self.jump_audio_root, "specs")
        return (
            os.path.join(spec_dir, base + ".npy"),
            os.path.join(spec_dir, base + ".png"),
        )

    def add_jump_window(self, start: int, end: int, label: str):
        start = min(start, end)
        end = max(start, end)
        if end == start:
            print("Jump window must span at least one frame.")
            return
        obj_id = None
        t0 = start / max(self.fps, 1e-6)
        t1 = end / max(self.fps, 1e-6)
        label_letter = {"snap": "s", "grunt": "g", "none": "n"}.get(label, "n")

        # Upsert by identical frame range
        existing_idx = next((i for i, w in enumerate(self.jump_windows) if w["start_frame"] == start and w["end_frame"] == end), None)
        entry = {
            "start_frame": start,
            "end_frame": end,
            "label": label,
            "label_letter": label_letter,
            "t0": t0,
            "t1": t1,
        }
        entry.update({"audio_path": None, "spec_npy_path": None, "spec_png_path": None})

        if existing_idx is not None:
            self.jump_windows[existing_idx] = entry
            self.selected_jump_index = existing_idx
        else:
            self.jump_windows.append(entry)
            self.selected_jump_index = len(self.jump_windows) - 1

        self.refresh_jump_timeline()
        self.mark_dirty()

    # ---------- Box propagation ----------
    def _selected_box_on_current_frame(self):
        """Return (obj_id, box_tuple) for the first scene-selected box on the current frame."""
        for item in self.scene.selectedItems():
            if isinstance(item, BoundingBoxItem):
                rect = item.mapRectToScene(item.rect())
                box_tuple = (
                    item.object_id,
                    item.class_id,
                    int(rect.left()),
                    int(rect.top()),
                    int(rect.right()),
                    int(rect.bottom()),
                    item.confidence,
                )
                return item.object_id, box_tuple
        return None, None

    def _propagate_box(self, obj_id: int, box_data, frame_count: int, direction: int, stop_on_existing: bool, overwrite_existing: bool):
        """Copy a box forward/backward across frames."""
        if frame_count <= 0 or direction == 0:
            return
        new_boxes_added = 0
        overwritten_count = 0
        for step in range(1, frame_count + 1):
            target_frame = self.current_frame_index + step * direction
            if target_frame < 0 or target_frame >= self.total_frames:
                break
            existing_idx = next((i for i, b in enumerate(self.annotations[target_frame]) if b[0] == obj_id), None)
            if existing_idx is not None:
                if overwrite_existing:
                    self.annotations[target_frame][existing_idx] = box_data
                    overwritten_count += 1
                    continue
                if stop_on_existing:
                    break
                continue
            self.annotations[target_frame].append(box_data)
            new_boxes_added += 1
        if new_boxes_added > 0 or overwritten_count > 0:
            self.mark_dirty()
            self.refresh_timeline()
            self.display_frame()
            print(
                f"Propagated box ID {obj_id} to {new_boxes_added} new frame(s)"
                + (f" and overwrote {overwritten_count} existing frame(s)." if overwritten_count else ".")
            )
        else:
            print("No frames updated during propagation.")

    def open_propagate_dialog(self):
        """Prompt for propagation options and copy the selected box."""
        obj_id, box = self._selected_box_on_current_frame()
        if obj_id is None or box is None:
            print("Select a box in the current frame to propagate.")
            return

        dialog = QDialog(self)
        dialog.setWindowTitle("Propagate Box")
        layout = QVBoxLayout(dialog)
        layout.addWidget(QLabel("Copy the selected box across frames to cover missed detections."))

        spin_layout = QHBoxLayout()
        spin_layout.addWidget(QLabel("Frames to copy:"))
        spin = QSpinBox()
        spin.setRange(1, max(1, self.total_frames - 1))
        spin.setValue(min(self.last_propagate_count, self.total_frames - 1))
        spin_layout.addWidget(spin)
        layout.addLayout(spin_layout)

        dir_layout = QHBoxLayout()
        dir_layout.addWidget(QLabel("Direction:"))
        dir_combo = QComboBox()
        dir_combo.addItems(["Forward", "Backward"])
        dir_layout.addWidget(dir_combo)
        layout.addLayout(dir_layout)

        stop_check = QCheckBox("Stop when an existing box for this ID is encountered")
        stop_check.setChecked(self.propagate_stop_on_existing)
        layout.addWidget(stop_check)

        overwrite_check = QCheckBox("Overwrite existing boxes for this ID")
        overwrite_check.setChecked(self.propagate_overwrite_existing)
        layout.addWidget(overwrite_check)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        layout.addWidget(buttons)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)

        if dialog.exec_() != QDialog.Accepted:
            return

        frame_count = spin.value()
        self.last_propagate_count = frame_count
        direction = 1 if dir_combo.currentText() == "Forward" else -1
        self.propagate_stop_on_existing = stop_check.isChecked()
        self.propagate_overwrite_existing = overwrite_check.isChecked()
        self._propagate_box(
            obj_id,
            box,
            frame_count,
            direction,
            self.propagate_stop_on_existing,
            self.propagate_overwrite_existing,
        )

    def load_jump_manifest(self):
        path = self._jump_manifest_path()
        if not os.path.exists(path):
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                repaired = []
                for win in data:
                    start = win.get("start_frame")
                    end = win.get("end_frame")
                    label = win.get("label", "none")
                    if start is None or end is None:
                        continue
                    t0 = start / max(self.fps, 1e-6)
                    t1 = end / max(self.fps, 1e-6)
                    repaired.append(
                        {
                            "start_frame": start,
                            "end_frame": end,
                            "label": label,
                            "label_letter": {"snap": "s", "grunt": "g", "none": "n"}.get(label, "n"),
                            "t0": t0,
                            "t1": t1,
                        }
                    )
                self.jump_windows = repaired
                self.refresh_jump_timeline()
                print(f"Loaded {len(self.jump_windows)} jump windows.")
        except Exception as exc:
            print(f"Failed to load jump manifest: {exc}")

    def save_jump_manifest(self):
        path = self._jump_manifest_path()
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.jump_windows, f, indent=2)
        except Exception as exc:
            print(f"Failed to save jump manifest: {exc}")

    # ---------- Playback ----------
    def _frame_to_ms(self, frame_idx: int) -> int:
        """Convert a frame index to milliseconds based on FPS."""
        return int((frame_idx / max(self.fps, 1e-6)) * 1000)

    def _frame_from_ms(self, position_ms: int) -> int:
        """Convert a timestamp in ms back to the nearest frame index."""
        frame = int(round((position_ms / 1000.0) * self.fps))
        return max(0, min(frame, self.total_frames - 1))

    def _sync_audio_to_current_frame(self):
        """Ensure the audio player is aligned to the current video frame."""
        position_ms = self._frame_to_ms(self.current_frame_index)
        self.audio_player.setPosition(position_ms)

    def _set_frame_from_playback(self, frame_idx: int):
        """Update the displayed frame during playback without extra saves."""
        frame_idx = max(0, min(frame_idx, self.total_frames - 1))
        if frame_idx == self.current_frame_index:
            return
        self.current_frame_index = frame_idx
        self.frame_slider.blockSignals(True)
        self.frame_slider.setValue(self.current_frame_index)
        self.frame_slider.blockSignals(False)
        self.display_frame()

    def toggle_playback(self, force_state=None):
        target_state = force_state if force_state is not None else not self.playing
        if target_state and self.total_frames <= 0:
            return
        if target_state:
            interval_ms = max(10, int(1000 / max(self.fps, 1e-6)))
            self.play_timer.start(interval_ms)
            # Sync audio to current frame position
            self._sync_audio_to_current_frame()
            self.audio_player.play()
            self.btn_play_pause.setText("Pause")
            self.playing = True
        else:
            self.play_timer.stop()
            self.audio_player.pause()
            self.btn_play_pause.setText("Play")
            self.playing = False

    def advance_playback(self):
        if not self.playing:
            return
        position_ms = self.audio_player.position()
        target_frame = self._frame_from_ms(position_ms)
        if target_frame != self.current_frame_index:
            self._set_frame_from_playback(target_frame)
        if target_frame >= self.total_frames - 1:
            self.toggle_playback(force_state=False)

    def delete_selected_jump_window(self):
        if self.selected_jump_index is None:
            return
        if self.selected_jump_index < 0 or self.selected_jump_index >= len(self.jump_windows):
            return
        del self.jump_windows[self.selected_jump_index]
        self.selected_jump_index = None
        self.refresh_jump_timeline()
        self.mark_dirty()

    def on_timeline_frame_selected(self, frame_idx: int):
        """Jump to a frame when clicking the timeline."""
        self.save_current_annotations()
        self.current_frame_index = frame_idx
        self.frame_slider.blockSignals(True)
        self.frame_slider.setValue(self.current_frame_index)
        self.frame_slider.blockSignals(False)
        self._sync_audio_to_current_frame()
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
            self._update_propagate_button_state()
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
            self._update_propagate_button_state()
                 
    def on_jump_timeline_frame_selected(self, frame_idx: int):
        """Jump to a frame when clicking the jump timeline."""
        self.on_timeline_frame_selected(frame_idx)

    def on_jump_window_selected(self, idx: int):
        self.selected_jump_index = idx
        self.jump_timeline_view.set_selected_index(idx)
        if 0 <= idx < len(self.jump_windows):
            current_label = self.jump_windows[idx]["label"]
            set_idx = self.jump_sound_combo.findText(current_label)
            if set_idx != -1:
                self.jump_sound_combo.blockSignals(True)
                self.jump_sound_combo.setCurrentIndex(set_idx)
                self.jump_sound_combo.blockSignals(False)
            self.jump_sound_combo.setEnabled(True)
        else:
            self.jump_sound_combo.blockSignals(True)
            self.jump_sound_combo.setCurrentIndex(self.jump_sound_combo.findText("none"))
            self.jump_sound_combo.blockSignals(False)
            self.jump_sound_combo.setEnabled(False)

    def set_selected_jump_sound(self):
        if self.selected_jump_index is None:
            self.jump_sound_combo.blockSignals(True)
            self.jump_sound_combo.setCurrentIndex(self.jump_sound_combo.findText("none"))
            self.jump_sound_combo.blockSignals(False)
            self.jump_sound_combo.setEnabled(False)
            return
        if not (0 <= self.selected_jump_index < len(self.jump_windows)):
            return
        label = self.jump_sound_combo.currentText()
        label_letter = {"snap": "s", "grunt": "g", "none": "n"}.get(label, "n")
        self.jump_windows[self.selected_jump_index]["label"] = label
        self.jump_windows[self.selected_jump_index]["label_letter"] = label_letter
        self.refresh_jump_timeline()
        self.mark_dirty()
        self.jump_timeline_view.setFocus(Qt.OtherFocusReason)

    def on_slider_changed(self, value):
        """Handle slider value change to navigate to a specific frame."""
        # Prevent recursive updates
        if value != self.current_frame_index:
            self.save_current_annotations()
            self.current_frame_index = value
            self._sync_audio_to_current_frame()
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
            self._sync_audio_to_current_frame()
            self.display_frame()
            print(f"Showing frame: {self.current_frame_index}")
        else:
            print("End of video reached.")
            if self.playing:
                self.toggle_playback(force_state=False)
    
    def go_to_previous_frame(self):
        self.save_current_annotations()
        if self.current_frame_index > 0:
            self.current_frame_index -= 1
            # Update slider without triggering valueChanged (to avoid recursion)
            self.frame_slider.blockSignals(True)
            self.frame_slider.setValue(self.current_frame_index)
            self.frame_slider.blockSignals(False)
            self._sync_audio_to_current_frame()
            self.display_frame()
            print(f"Showing frame: {self.current_frame_index}")
        else:
            print('No previous frame')
    
    def delete_selected_boxes(self):
        """Remove selected bounding boxes from the scene and stored annotations."""
        selected_items = list(self.scene.selectedItems())
        for item in selected_items:
            if isinstance(item, BoundingBoxItem):
                item_scene = item.scene()
                if item_scene is not None:
                    item_scene.removeItem(item)
        self.save_current_annotations()
        self.display_frame()
        self.refresh_timeline()

    def on_scene_selection_changed(self):
        """Keep selection-driven UI (including propagation) in sync with scene picks."""
        selected_items = [i for i in self.scene.selectedItems() if isinstance(i, BoundingBoxItem)]
        selected_ids = {i.object_id for i in selected_items if i.object_id is not None}
        if selected_ids != self.selected_object_ids:
            self.selected_object_ids = selected_ids
            self.sync_current_frame_objects_list()
            self.timeline_view.set_selected_object_ids(self.selected_object_ids)
            has_selection = len(self.selected_object_ids) > 0
            self.merge_button.setEnabled(len(self.selected_object_ids) > 1)
            self.delete_object_button.setEnabled(has_selection)
            self.delete_id_global_button.setEnabled(has_selection)
        self._update_propagate_button_state()

    def export_annotated_video(self, output_video_path: str):
        """Save a video with the current annotations drawn on each frame."""
        export_cap = cv2.VideoCapture(self.video_path)
        if not export_cap.isOpened():
            print(f"Could not reopen video for export: {self.video_path}")
            return

        fps = export_cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(export_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(export_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        if not writer.isOpened():
            print(f"Could not open writer for {output_video_path}")
            export_cap.release()
            return

        for idx in range(self.total_frames):
            ret, frame = export_cap.read()
            if not ret:
                print(f"Stopped early at frame {idx} while exporting video.")
                break

            for obj_id, _, x_min, y_min, x_max, y_max, _ in self.annotations[idx]:
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                label = f"ID {obj_id}" if obj_id is not None else "ID ?"
                cv2.putText(
                    frame,
                    label,
                    (x_min, max(0, y_min - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )

            writer.write(frame)

        export_cap.release()
        writer.release()
        print(f"Saved annotated video to {output_video_path}")

    def save_dataset(self):
        """Export frames and labels in YOLO format to annotations/data/<video_name> or test/data/<video_name>."""
        self.save_current_annotations()
        video_stem = self.dataset_stem
        output_dir = os.path.join(DATASET_DIR, video_stem)
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
            for obj_id, class_id, x_min, y_min, x_max, y_max, _ in self.annotations[idx]:
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

        if self.save_video:
            annotated_video_path = os.path.join(output_dir, f"{video_stem}_annotated.mp4")
            self.export_annotated_video(annotated_video_path)

        # Record which snippet/full video these labels correspond to for future reloads
        if self.write_snippet_manifest:
            save_snippet_manifest(video_stem, self.video_path, self.source_video_path)

        # Persist jump windows manifest on explicit save
        self.save_jump_manifest()

        self.mark_clean()

    def keyPressEvent(self, event):        
        if event.key() == Qt.Key_J:
            self.mark_jump_start()
            event.accept()
            return
        if event.key() == Qt.Key_K:
            self.mark_jump_end()
            event.accept()
            return
        if event.key() == Qt.Key_Space:
            self.toggle_playback()
            event.accept()
            return
        if event.key() == Qt.Key_P:
            self.open_propagate_dialog()
            event.accept()
            return
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
            elif self.selected_jump_index is not None:
                self.delete_selected_jump_window()
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
        for obj_id, class_id, x_min, y_min, x_max, y_max, _ in boxes:
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
        self.jump_timeline_view.set_current_frame(self.current_frame_index)
    
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
        self._update_propagate_button_state()
        self.update_merge_all_button_state()
    
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
        self.update_merge_all_button_state()
    
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
            for i, (obj_id, class_id, x_min, y_min, x_max, y_max, confidence) in enumerate(self.annotations[frame_idx]):
                if obj_id in ids_to_merge:
                    # Replace with target ID
                    self.annotations[frame_idx][i] = (target_id, class_id, x_min, y_min, x_max, y_max, confidence)
        
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
        self.update_merge_all_button_state()
        
        print(f"Merge complete. All objects now use ID {target_id}")
        # Return focus so Left/Right arrows keep working without reselecting
        self.timeline_view.setFocus(Qt.OtherFocusReason)

    def merge_all_objects(self):
        """Merge every object ID into a single timeline (lowest ID wins)."""
        if len(self.all_object_ids) < 2:
            return
        confirm_box = QMessageBox(self)
        confirm_box.setWindowTitle("Merge All Objects")
        confirm_box.setIcon(QMessageBox.NoIcon)
        confirm_box.setText("Are you sure you want to merge all objects?")
        confirm_box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        confirm_box.setDefaultButton(QMessageBox.No)
        if confirm_box.exec_() != QMessageBox.Yes:
            return

        target_id = min(self.all_object_ids)
        ids_to_merge = set(self.all_object_ids) - {target_id}
        print(f"Merging all object IDs {ids_to_merge} into {target_id}")

        for frame_idx in range(self.total_frames):
            for i, (obj_id, class_id, x_min, y_min, x_max, y_max, confidence) in enumerate(self.annotations[frame_idx]):
                if obj_id in ids_to_merge:
                    self.annotations[frame_idx][i] = (target_id, class_id, x_min, y_min, x_max, y_max, confidence)

        self.all_object_ids = {target_id}
        self.selected_object_ids.clear()
        self.merge_button.setEnabled(False)
        self.delete_object_button.setEnabled(False)
        self.delete_id_global_button.setEnabled(False)
        self.timeline_view.set_selected_object_ids(self.selected_object_ids)

        self.display_frame()
        self.refresh_timeline()
        self.mark_dirty()
        self.update_merge_all_button_state()
        print(f"Merge-all complete. All objects now use ID {target_id}")
        self.timeline_view.setFocus(Qt.OtherFocusReason)
    
    def delete_selected_objects_from_frame(self):
        """Delete all bounding boxes with selected object IDs from the current frame only."""
        if len(self.selected_object_ids) == 0:
            return
        
        # Filter out annotations with selected object IDs
        original_count = len(self.annotations[self.current_frame_index])
        self.annotations[self.current_frame_index] = [
            (obj_id, class_id, x_min, y_min, x_max, y_max, confidence)
            for obj_id, class_id, x_min, y_min, x_max, y_max, confidence in self.annotations[self.current_frame_index]
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
                (obj_id, class_id, x_min, y_min, x_max, y_max, confidence)
                for obj_id, class_id, x_min, y_min, x_max, y_max, confidence in self.annotations[frame_idx]
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
                    (oid, cls, x1, y1, x2, y2, conf)
                    for (oid, cls, x1, y1, x2, y2, conf) in self.annotations[frame_idx]
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
        '--video-path',
        type=str, 
        help='The path to the local MP4 video file to annotate.'
    )
    source_group.add_argument(
        '--box-video',
        type=str,
        help='Exact Box filename to download and annotate (e.g., video.mp4).'
    )
    # Added argument for YOLO weights
    parser.add_argument(
        '--weights',
        type=str, 
        required=True, 
        help='The path to the YOLO model weights file (.pt).'
    )
    parser.add_argument(
        '--save_video',
        action='store_true',
        help='Also save an annotated video with bounding boxes when saving the dataset.',
    )
    parser.add_argument(
        '--run-new',
        action='store_true',
        help='Force re-running model inference and ignore any cached labels for this video.',
    )
    parser.add_argument(
        '--base-dir',
        type=str,
        default='.',
        help='Project root (controls annotations/, test/, and system_files/).',
    )
    parser.add_argument(
        '--snippet',
        action='store_true',
        help='Create and annotate a ~5s snippet instead of the full video when no cached labels exist.',
    )
    split_group = parser.add_mutually_exclusive_group()
    split_group.add_argument(
        '--test_set',
        action='store_true',
        help='Save annotations under test/ (default).',
    )
    split_group.add_argument(
        '--train_set',
        action='store_true',
        help='Save annotations under annotations/.',
    )
    args = parser.parse_args()
    configure_paths(args.base_dir)
    global OUTPUT_ROOT, DATASET_DIR, VIDEO_DIR
    OUTPUT_ROOT = ANNOTATIONS_ROOT if args.train_set else TEST_ROOT
    DATASET_DIR = os.path.join(OUTPUT_ROOT, "data")
    VIDEO_DIR = os.path.join(OUTPUT_ROOT, "videos")
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    app = QApplication(sys.argv)
    
    if args.video_path:
        candidate_name = os.path.basename(args.video_path)
    else:
        candidate_name = args.box_video
    dataset_stem = os.path.splitext(candidate_name)[0]
    other_root = ANNOTATIONS_ROOT if OUTPUT_ROOT == TEST_ROOT else TEST_ROOT
    if not confirm_cross_split(dataset_stem, candidate_name, other_root):
        print("Aborted due to annotations/test conflict.")
        sys.exit(1)
    try:
        resolved_video_path = resolve_video_path(args.video_path, args.box_video)
    except (FileNotFoundError, ValueError) as exc:
        print(exc)
        sys.exit(1)
    cached_labels_dir = os.path.join(DATASET_DIR, dataset_stem, "labels")
    has_cached_labels = (not args.run_new) and os.path.isdir(cached_labels_dir) and any(
        f.endswith(".txt") for f in os.listdir(cached_labels_dir)
    )

    snippet_from_manifest = load_snippet_from_manifest(dataset_stem) if has_cached_labels else None
    snippet_guess = find_existing_snippet(dataset_stem) if has_cached_labels and snippet_from_manifest is None else None

    if has_cached_labels:
        if snippet_from_manifest:
            print(f"Cached labels found; using snippet from manifest: {snippet_from_manifest}")
            video_to_use = snippet_from_manifest
        elif snippet_guess:
            print(f"Cached labels found; using latest snippet for {dataset_stem}: {snippet_guess}")
            video_to_use = snippet_guess
        else:
            print(f"Cached labels found at {cached_labels_dir}, but no snippet located; using original video.")
            video_to_use = resolved_video_path
    else:
        if args.snippet:
            video_to_use = create_random_snippet(resolved_video_path, duration_seconds=5.0)
        else:
            print("No cached labels; processing full video (no snippet requested).")
            video_to_use = resolved_video_path

    window = MainWindow(
        video_path=video_to_use,
        yolo_weights=args.weights,
        save_video=args.save_video,
        save_clips=False,
        run_new=args.run_new,
        dataset_stem=dataset_stem,
        source_video_path=resolved_video_path,
        write_snippet_manifest=args.snippet,
    ) 
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
