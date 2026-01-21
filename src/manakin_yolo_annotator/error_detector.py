import argparse
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set

import cv2
import numpy as np
from PyQt5.QtCore import Qt, QRectF, QPoint, QPointF, pyqtSignal, QTimer
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QKeySequence, QFont, QFontMetrics, QPalette, QColor, QBrush, QPolygonF
from PyQt5.QtWidgets import (
    QApplication,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QShortcut,
    QSizePolicy,
    QFrame,
)
import sip
from ultralytics import YOLO

# Assumed relative imports based on your command structure
from .bounding_box import BoundingBoxItem, generate_color_for_id
from .box_downloader import BoxNavigator
from .random_vid import select_random_box_video_names, DEFAULT_BOX_FOLDER_ID


def iou(box1: np.ndarray, box2: np.ndarray) -> float:
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])
    inter_w = max(0.0, xB - xA)
    inter_h = max(0.0, yB - yA)
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return float(inter / union) if union > 0 else 0.0


def near_edge(box: np.ndarray, width: int, height: int, margin_frac: float = 0.05) -> bool:
    margin_w = width * margin_frac
    margin_h = height * margin_frac
    x1, y1, x2, y2 = box
    return (x1 <= margin_w) or (y1 <= margin_h) or ((width - x2) <= margin_w) or ((height - y2) <= margin_h)


class VideoAnalysisResult:
    def __init__(
        self,
        frames: List[np.ndarray],
        boxes_per_frame: List[List[List[float]]],
        reasons: List[str],
        flagged_frames: List[int],
        frame_reasons: Dict[int, List[str]],
        fps: float,
    ):
        self.frames = frames
        self.boxes_per_frame = boxes_per_frame
        self.reasons = reasons
        self.flagged_frames = flagged_frames
        self.frame_reasons = frame_reasons
        self.fps = fps


def analyze_video(
    video_path: Path,
    model: YOLO,
    iou_threshold: float = 0.1,
    flicker_len: int = 3,
    edge_margin: float = 0.05,
    force_return: bool = False,
) -> Optional[VideoAnalysisResult]:
    frames: List[np.ndarray] = []
    boxes_per_frame: List[List[List[float]]] = []
    reasons: List[str] = []
    flagged_frames: Set[int] = set()
    frame_reasons: Dict[int, Set[str]] = {}

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[error] Could not open video {video_path}")
        return None
    if is_grayscale_video(cap):
        print(f"[info] Skipping grayscale video (no inference): {video_path}")
        cap.release()
        return None
        
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    cap.release()

    frame_idx = 0
    for result in model.track(str(video_path), stream=True, persist=True, verbose=True, iou=0.2):
        img = result.orig_img
        frames.append(img.copy())
        
        frame_boxes: List[List[float]] = []
        boxes_tensor = result.boxes
        if boxes_tensor is not None and len(boxes_tensor) > 0:
            xyxy = boxes_tensor.xyxy.cpu().numpy()
            cls = boxes_tensor.cls.cpu().numpy()
            confs = boxes_tensor.conf.cpu().numpy()
            
            for i in range(len(xyxy)):
                x1, y1, x2, y2 = xyxy[i]
                c = int(cls[i])
                conf = float(confs[i])
                frame_boxes.append([x1, y1, x2, y2, c, conf])

            if len(frame_boxes) >= 2:
                arr = np.array([b[:4] for b in frame_boxes], dtype=float)
                is_multi_distinct = False
                for i in range(len(arr)):
                    for j in range(i + 1, len(arr)):
                        if iou(arr[i], arr[j]) < iou_threshold:
                            is_multi_distinct = True
                            break
                    if is_multi_distinct:
                        break
                
                if is_multi_distinct:
                    flagged_frames.add(frame_idx)
                    frame_reasons.setdefault(frame_idx, set()).add("Multiple Non-Overlapping Boxes")
                    if "Multiple Non-Overlapping Boxes" not in reasons:
                        reasons.append("Multiple Non-Overlapping Boxes")

        boxes_per_frame.append(frame_boxes)
        frame_idx += 1

    total_frames = len(boxes_per_frame)
    if total_frames == 0:
        return None

    presence = [len(b) > 0 for b in boxes_per_frame]
    runs = []
    if total_frames > 0:
        current_val = presence[0]
        run_start = 0
        for idx in range(1, total_frames):
            if presence[idx] != current_val:
                runs.append((current_val, run_start, idx - 1))
                run_start = idx
                current_val = presence[idx]
        runs.append((current_val, run_start, total_frames - 1))

    valid_runs = []
    for val, s, e in runs:
        run_len = e - s + 1
        if val:
            if run_len <= flicker_len:
                reasons.append(f"Flicker (Short Detection)")
                flagged_frames.update(range(s, e + 1))
                for f in range(s, e + 1):
                    frame_reasons.setdefault(f, set()).add(f"Flicker (Short Detection: {run_len}f)")
            else:
                valid_runs.append((s, e))
        else:
            if s > 0 and e < total_frames - 1:
                if run_len <= flicker_len:
                    reasons.append(f"Flicker (Dropout)")
                    flagged_frames.update(range(s, e + 1))
                    for f in range(s, e + 1):
                        frame_reasons.setdefault(f, set()).add(f"Flicker (Dropout: {run_len}f)")

    def get_largest_box(idx):
        bxs = boxes_per_frame[idx]
        if not bxs: return None
        return max(bxs, key=lambda b: (b[2]-b[0])*(b[3]-b[1]))[:4]

    for s, e in valid_runs:
        if s > 0: 
            box = get_largest_box(s)
            if box is not None and not near_edge(np.array(box), width, height, edge_margin):
                reasons.append("Sudden Appearance (Center)")
                flagged_frames.add(s)
                frame_reasons.setdefault(s, set()).add("Sudden Appearance (Center)")
        if e < total_frames - 1:
            box = get_largest_box(e)
            if box is not None and not near_edge(np.array(box), width, height, edge_margin):
                reasons.append("Sudden Disappearance (Center)")
                flagged_frames.add(e)
                frame_reasons.setdefault(e, set()).add("Sudden Disappearance (Center)")

    if not reasons and not force_return:
        return None

    unique_reasons = sorted(list(set(reasons)))
    sorted_flagged = sorted(list(flagged_frames))
    sorted_frame_reasons = {k: sorted(list(v)) for k, v in frame_reasons.items()}

    return VideoAnalysisResult(
        frames=frames,
        boxes_per_frame=boxes_per_frame,
        reasons=unique_reasons,
        flagged_frames=sorted_flagged,
        frame_reasons=sorted_frame_reasons,
        fps=fps,
    )


def qpixmap_from_bgr(image: np.ndarray) -> QPixmap:
    h, w, _ = image.shape
    bytes_per_line = 3 * w
    qimg = QImage(image.data, w, h, bytes_per_line, QImage.Format_BGR888)
    return QPixmap.fromImage(qimg.copy())


def is_grayscale_video(
    cap: cv2.VideoCapture,
    sample_count: int = 5,
    diff_threshold: float = 0.5,
) -> bool:
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total <= 0:
        return False

    indices = sorted(set(
        int(i * (total - 1) / max(sample_count - 1, 1))
        for i in range(sample_count)
    ))

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        if frame.ndim < 3 or frame.shape[2] < 3:
            continue
        b = frame[:, :, 0].astype(np.int16)
        g = frame[:, :, 1].astype(np.int16)
        r = frame[:, :, 2].astype(np.int16)
        diff = (np.mean(np.abs(b - g)) + np.mean(np.abs(b - r)) + np.mean(np.abs(g - r))) / 3.0
        if diff > diff_threshold:
            return False

    return True


class SimpleView(QGraphicsView):
    def __init__(self, scene: QGraphicsScene, on_new_box, parent=None):
        super().__init__(scene, parent)
        self._dragging = False
        self._drag_start = None
        self._rubber_band_item = None
        self.on_new_box = on_new_box
        self.setRenderHint(QPainter.Antialiasing)
        self.setMouseTracking(True)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self._user_zoomed = False

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            item = self.itemAt(event.pos())
            if item and not isinstance(item, QGraphicsPixmapItem):
                super().mousePressEvent(event)
                return
            self._dragging = True
            self._drag_start = self.mapToScene(event.pos())
            self._clear_rubber_band()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._dragging:
            current = self.mapToScene(event.pos())
            rect = QRectF(self._drag_start, current).normalized()
            if self._rubber_band_item and sip.isdeleted(self._rubber_band_item):
                self._rubber_band_item = None
            if self._rubber_band_item is None:
                pen = QPen(self.palette().highlight().color())
                pen.setWidth(1)
                self._rubber_band_item = self.scene().addRect(rect, pen=pen)
            else:
                self._rubber_band_item.setRect(rect)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._dragging and event.button() == Qt.LeftButton:
            self._dragging = False
            current = self.mapToScene(event.pos())
            rect = QRectF(self._drag_start, current).normalized()
            self._clear_rubber_band()
            if rect.width() > 4 and rect.height() > 4:
                self.on_new_box(rect)
        super().mouseReleaseEvent(event)

    def _clear_rubber_band(self):
        if not self._rubber_band_item:
            return
        try:
            if not sip.isdeleted(self._rubber_band_item):
                scene = self.scene()
                if scene is not None:
                    scene.removeItem(self._rubber_band_item)
        except RuntimeError:
            pass
        self._rubber_band_item = None

    def fit_content(self):
        if self.scene() is None:
            return
        self.resetTransform()
        self.fitInView(self.sceneRect(), Qt.KeepAspectRatio)
        self._user_zoomed = False

    def zoom_in(self):
        self.scale(1.25, 1.25)
        self._user_zoomed = True

    def zoom_out(self):
        self.scale(0.8, 0.8)
        self._user_zoomed = True


class BadTimelineView(QGraphicsView):
    frame_selected = pyqtSignal(int)

    def __init__(self, total_frames: int, parent=None):
        self.scene_obj = QGraphicsScene(parent)
        super().__init__(self.scene_obj, parent)
        self.total_frames = max(total_frames, 1)
        fm = QFontMetrics(QFont())
        self.left_margin = max(70, fm.boundingRect("Presence").width() + 12)
        self.base_width = 800
        self.timeline_width = self.base_width
        self.bar_height = 10 
        self.row_spacing = 4
        self.setMinimumHeight(100)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setFrameStyle(QFrame.NoFrame)
        self.setFocusPolicy(Qt.StrongFocus)
        self.presence = {}
        self.flagged_frames = set()
        self.saved_frames = set()
        self.zoom_factor = 1.0
        self.current_frame = 0
        self.marker_handle_rect = None
        self.marker_bar_rect = None
        self.dragging_marker = False
        self._redraw()

    def set_total_frames(self, total_frames: int):
        self.total_frames = max(total_frames, 1)
        self._redraw()

    def set_presence(self, presence: dict):
        self.presence = {k: set(v) for k, v in presence.items()}
        self._redraw()

    def set_flagged_frames(self, frames: List[int]):
        self.flagged_frames = set(frames)
        self._redraw()

    def set_saved_frames(self, frames: List[int]):
        self.saved_frames = set(frames)
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
        fm = QFontMetrics(QFont())
        label_width = fm.boundingRect("Presence").width() + 12
        self.left_margin = max(70, label_width)
        self.base_width = max(200, self.viewport().width() - self.left_margin - 20)
        self.timeline_width = max(200, self.base_width * self.zoom_factor)
        scale = self.timeline_width / float(max(self.total_frames - 1, 1))
        y = 10

        frames = sorted(self.presence.get(0, []))
        if frames:
            color = QColor(100, 200, 100)
            label_item = self.scene_obj.addText("Presence", QFont())
            label_item.setDefaultTextColor(color.darker(150))
            label_item.setPos(4, y - 2)
            
            start = frames[0]
            end = frames[0]
            for f in frames[1:] + [None]:
                if f is not None and f == end + 1:
                    end = f
                    continue
                x_start = self.left_margin + start * scale
                width = max(2, (end - start + 1) * scale)
                rect = QRectF(x_start, y, width, self.bar_height)
                pen = QPen(color.darker(120), 1)
                brush = QBrush(color)
                self.scene_obj.addRect(rect, pen, brush)
                start = f
                end = f
            y += self.bar_height + self.row_spacing
        else:
             label_item = self.scene_obj.addText("No Detections", QFont())
             label_item.setDefaultTextColor(QColor(150, 150, 150))
             label_item.setPos(4, y - 2)
             y += self.bar_height + self.row_spacing

        total_height = y + 20
        marker_x = self.left_margin + self.current_frame * scale
        marker_pen = QPen(QColor(90, 90, 90, 180), 1.5, Qt.SolidLine)
        self.scene_obj.addLine(marker_x, 0, marker_x, total_height, marker_pen)
        
        handle_width = 10
        handle_height = total_height
        self.marker_handle_rect = QRectF(marker_x - handle_width / 2, 0, handle_width, handle_height)
        handle_brush = QBrush(QColor(140, 140, 140, 60))
        self.scene_obj.addRect(self.marker_handle_rect, QPen(Qt.NoPen), handle_brush)
        
        bar_height_scrub = 6
        self.marker_bar_rect = QRectF(self.left_margin, 0, self.timeline_width, bar_height_scrub)
        bar_pen = QPen(QColor(110, 110, 110, 160), 1.2)
        bar_brush = QBrush(QColor(170, 170, 170, 120))
        self.scene_obj.addRect(self.marker_bar_rect, bar_pen, bar_brush)

        if self.saved_frames:
            saved_pen = QPen(QColor(46, 204, 113), 0)
            saved_brush = QBrush(QColor(46, 204, 113))
            tick_h = 4
            tick_y = -tick_h - 1
            for f in self.saved_frames:
                if f < 0 or f >= self.total_frames:
                    continue
                x = self.left_margin + f * scale
                tick_w = max(2, scale)
                rect = QRectF(x, tick_y, tick_w, tick_h)
                self.scene_obj.addRect(rect, saved_pen, saved_brush)

        if self.flagged_frames:
            red_pen = QPen(QColor(200, 0, 0), 0)
            red_brush = QBrush(QColor(200, 0, 0))
            tick_h = 4
            tick_y = (bar_height_scrub - tick_h) / 2.0
            for f in self.flagged_frames:
                if f < 0 or f >= self.total_frames:
                    continue
                x = self.left_margin + f * scale
                tick_w = max(2, scale)
                rect = QRectF(x, tick_y, tick_w, tick_h)
                self.scene_obj.addRect(rect, red_pen, red_brush)

        self.scene_obj.setSceneRect(0, -8, self.left_margin + self.timeline_width, total_height + 13)

    def _frame_from_x(self, x: float) -> int:
        scale = self.timeline_width / float(max(self.total_frames - 1, 1))
        frame = int(round((x - self.left_margin) / scale))
        return max(0, min(frame, self.total_frames - 1))

    def mousePressEvent(self, event):
        scene_pos = self.mapToScene(event.pos())
        if self.marker_bar_rect and self.marker_bar_rect.contains(scene_pos) and event.button() == Qt.LeftButton:
            frame = self._frame_from_x(scene_pos.x())
            self.set_current_frame(frame)
            self.frame_selected.emit(frame)
            self.dragging_marker = True
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
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
        if self.dragging_marker and event.button() == Qt.LeftButton:
            self.dragging_marker = False
            event.accept()
            return
        super().mouseReleaseEvent(event)


class Error_Detector(QMainWindow):
    def __init__(
        self,
        model: YOLO,
        navigator: BoxNavigator,
        candidates: List[str],
        output_dir: Path,
        iou_threshold: float = 0.1,
        flicker_len: int = 3,
        forced_video: Optional[Path] = None,
        run_new: bool = False,
        other_split_root: Optional[Path] = None,
    ):
        super().__init__()
        self.model = model
        self.navigator = navigator
        self.candidates = candidates
        random.shuffle(self.candidates)
        self.current_candidate_idx = 0
        self.forced_video = forced_video
        self.forced_consumed = False
        self.analysis: Optional[VideoAnalysisResult] = None
        self.frames: List[np.ndarray] = []
        self.boxes_per_frame: List[List[List[float]]] = []
        self.reasons: List[str] = []
        self.flagged_frames: List[int] = []
        self.frame_idx = 0
        self.output_dir = output_dir
        self.iou_threshold = iou_threshold
        self.flicker_len = flicker_len
        self.run_new = run_new
        self.other_split_root = other_split_root
        self.video_path: Optional[Path] = None
        self.fps = 30.0
        self.playing = False
        self.saved_state: Dict[int, List[List[float]]] = {}

        self.init_ui()
        
        self._shortcut_next = QShortcut(QKeySequence(Qt.Key_Right), self)
        self._shortcut_next.activated.connect(self.next_frame)
        self._shortcut_prev = QShortcut(QKeySequence(Qt.Key_Left), self)
        self._shortcut_prev.activated.connect(self.prev_frame)
        self._shortcut_del = QShortcut(QKeySequence(Qt.Key_Delete), self)
        self._shortcut_del.activated.connect(self.delete_selected_box)
        self._shortcut_backspace = QShortcut(QKeySequence(Qt.Key_Backspace), self)
        self._shortcut_backspace.activated.connect(self.delete_selected_box)
        self._shortcut_space = QShortcut(QKeySequence(Qt.Key_Space), self)
        self._shortcut_space.activated.connect(self.toggle_playback)

        self.find_and_load_flagged_video()

    def init_ui(self):
        self.setWindowTitle("Error_Detector")
        self.resize(1450, 950)
        main_layout = QVBoxLayout()
        
        # --- TOP HEADER: REORGANIZED STATS ---
        header_widget = QFrame()
        header_widget.setFrameShape(QFrame.StyledPanel)
        header_widget.setAutoFillBackground(True)
        header_widget.setStyleSheet("border-radius: 5px;")
        header_layout = QHBoxLayout(header_widget)
        header_layout.setContentsMargins(8, 6, 8, 6)
        header_layout.setSpacing(12)
        
        # Larger font settings
        primary_font = QFont("Arial", 15, QFont.Bold)
        normal_font = QFont("Arial", 15)
        info_font = QFont("Arial", 15)

        # 1. Video Name (Not bold, takes good space)
        info_vbox = QVBoxLayout()
        info_vbox.setSpacing(2)
        self.title_label = QLabel("Video: --")
        self.title_label.setFont(info_font)
        self.frame_label = QLabel("Frame: 0 / 0")
        self.frame_label.setFont(info_font)
        info_vbox.addWidget(self.title_label)
        info_vbox.addWidget(self.frame_label)
        
        # 2. Stats Column (Bold Boxes)
        saved_vbox = QVBoxLayout()
        saved_vbox.setSpacing(2)
        self.saved_counter_label = QLabel("Saved: 0")
        self.saved_counter_label.setFont(normal_font)
        saved_vbox.addWidget(self.saved_counter_label)

        boxes_vbox = QVBoxLayout()
        boxes_vbox.setSpacing(2)
        self.box_count_label = QLabel("Boxes: 0")
        self.box_count_label.setFont(primary_font) # Bold
        boxes_vbox.addWidget(self.box_count_label)
        
        # 4. Reason (Bold/Red when flagged, Large text)
        self.reason_label = QLabel("Flag: None")
        self.reason_label.setFont(primary_font)
        self.reason_label.setAlignment(Qt.AlignCenter)
        
        # 5. Buttons
        btn_vbox = QVBoxLayout()
        btn_vbox.setSpacing(4)
        self.btn_save_frame = QPushButton("SAVE FRAME")
        self.btn_save_frame.setMinimumSize(140, 36)
        self.btn_save_frame.setStyleSheet(self._button_outline_style())
        self.btn_save_frame.clicked.connect(self.save_current_frame)
        
        btn_next_vid = QPushButton("NEXT VIDEO")
        btn_next_vid.setMinimumSize(140, 30)
        btn_next_vid.setStyleSheet(self._button_outline_style())
        btn_next_vid.clicked.connect(self.next_video)
        btn_vbox.addWidget(self.btn_save_frame)
        btn_vbox.addWidget(btn_next_vid)
        
        header_layout.addLayout(info_vbox)
        header_layout.addSpacing(8)
        header_layout.addLayout(saved_vbox)
        header_layout.addSpacing(8)
        header_layout.addLayout(boxes_vbox)
        header_layout.addSpacing(12)
        header_layout.addWidget(self.reason_label)
        header_layout.addSpacing(12)
        header_layout.addLayout(btn_vbox)
        main_layout.addWidget(header_widget)
        
        # --- VIDEO VIEW ---
        self.scene = QGraphicsScene(self)
        self.view = SimpleView(self.scene, self.add_box)
        main_layout.addWidget(self.view, 10)
        
        # --- NAVIGATION ---
        nav_container = QWidget()
        nav_layout = QHBoxLayout(nav_container)
        
        btn_prev = QPushButton("◀ Prev")
        btn_prev.clicked.connect(self.prev_frame)
        btn_next = QPushButton("Next ▶")
        btn_next.clicked.connect(self.next_frame)
        
        self.play_button = QPushButton("Play")
        self.play_button.setFixedWidth(100)
        self.play_button.clicked.connect(self.toggle_playback)
        self.play_timer = QTimer()
        self.play_timer.timeout.connect(self.advance_playback)
        
        btn_del = QPushButton("Delete Box")
        btn_del.clicked.connect(self.delete_selected_box)
        
        btn_zoom_out = QPushButton("Zoom -")
        btn_zoom_out.clicked.connect(self.view.zoom_out)
        btn_zoom_in = QPushButton("Zoom +")
        btn_zoom_in.clicked.connect(self.view.zoom_in)
        btn_fit = QPushButton("Fit View")
        btn_fit.clicked.connect(self.view.fit_content)
        
        nav_layout.addWidget(btn_prev)
        nav_layout.addWidget(self.play_button)
        nav_layout.addWidget(btn_next)
        nav_layout.addSpacing(20)
        nav_layout.addWidget(btn_del)
        nav_layout.addStretch()
        nav_layout.addWidget(btn_zoom_out)
        nav_layout.addWidget(btn_zoom_in)
        nav_layout.addWidget(btn_fit)
        main_layout.addWidget(nav_container)
        
        # --- TIMELINE ---
        self.timeline_view = BadTimelineView(total_frames=1)
        self.timeline_view.frame_selected.connect(self.on_timeline_frame_selected)
        main_layout.addWidget(QLabel("Timeline (Red = Flagged)"))
        main_layout.addWidget(self.timeline_view)
        
        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

    def find_and_load_flagged_video(self):
        if self.forced_video and not self.forced_consumed:
            path = self.forced_video
            if not self._confirm_cross_split(path.name):
                self.forced_consumed = True
                print("Aborted due to annotations/test conflict.")
                raise SystemExit(1)
            if not path.exists():
                downloaded = self.navigator.download_vid(path.name)
                if downloaded: path = Path(downloaded)
                else:
                    QMessageBox.warning(self, "Error", f"Could not find {path}")
                    self.forced_consumed = True
                    return
            analysis = analyze_video(
                path,
                self.model,
                iou_threshold=self.iou_threshold,
                flicker_len=self.flicker_len,
                force_return=True,
            )
            if analysis: self.load_analysis(str(path), analysis)
            self.forced_consumed = True
            return

        while self.current_candidate_idx < len(self.candidates):
            name = self.candidates[self.current_candidate_idx]
            self.current_candidate_idx += 1
            if not self._confirm_cross_split(name):
                print("Aborted due to annotations/test conflict.")
                raise SystemExit(1)
            path = self.navigator.download_vid(name)
            if path is None: continue
            analysis = analyze_video(
                Path(path),
                self.model,
                iou_threshold=self.iou_threshold,
                flicker_len=self.flicker_len,
            )
            if analysis:
                self.load_analysis(path, analysis)
                return
        QMessageBox.information(self, "Done", "No more flagged videos found.")
        self.close()

    def load_analysis(self, video_path: str, analysis: VideoAnalysisResult):
        self.video_path = Path(video_path)
        self.analysis = analysis
        self.frames = analysis.frames
        self.boxes_per_frame = analysis.boxes_per_frame
        self.reasons = analysis.reasons
        self.flagged_frames = analysis.flagged_frames
        self.fps = max(1e-6, float(analysis.fps))
        self.frame_idx = 0
        self.saved_state.clear()
        if not self.run_new:
            self._load_saved_annotations()
        self._update_saved_counter()
        self.timeline_view.set_total_frames(len(self.frames))
        self.timeline_view.set_presence(self._build_presence())
        self.timeline_view.set_flagged_frames(self.flagged_frames)
        self.timeline_view.set_saved_frames(list(self.saved_state.keys()))
        self.update_view()

    def update_view(self):
        if not self.frames: return
        frame = self.frames[self.frame_idx]
        h, w, _ = frame.shape
        self.scene.clear()
        self.image_item = QGraphicsPixmapItem(qpixmap_from_bgr(frame))
        self.scene.addItem(self.image_item)
        self.scene.setSceneRect(0, 0, w, h)

        for box in self.boxes_per_frame[self.frame_idx]:
            x1, y1, x2, y2, cls, conf = box
            rect = QRectF(x1, y1, x2 - x1, y2 - y1)
            item = BoundingBoxItem(rect, class_id=cls, object_id=None, confidence=conf, on_change=self.on_box_change)
            self.scene.addItem(item)

        self.title_label.setText(f"Video: {self.video_path.name}")
        self.frame_label.setText(f"Frame {self.frame_idx + 1} / {len(self.frames)}")
        self.box_count_label.setText(f"Boxes: {len(self.boxes_per_frame[self.frame_idx])}")
        
        # PER-FRAME REASONING (Red and Bold only when flag exists)
        reasons_here = self.analysis.frame_reasons.get(self.frame_idx, [])
        if reasons_here:
            self.reason_label.setText(f"FLAG: {', '.join(reasons_here)}")
            self.reason_label.setStyleSheet("color: #d63031; font-weight: bold;")
        else:
            self.reason_label.setText("Flag: None")
            self.reason_label.setStyleSheet("color: #636e72; font-weight: normal;")

        # Save Frame Button Styling
        if self.frame_idx in self.saved_state:
            self.btn_save_frame.setText("FRAME SAVED ✓")
            self._set_save_button_saved_state(True)
        else:
            self.btn_save_frame.setText("SAVE FRAME")
            self._set_save_button_saved_state(False)
            
        if not self.view._user_zoomed:
            self.view.fit_content()
        self.timeline_view.set_current_frame(self.frame_idx)

    def on_box_change(self, box: BoundingBoxItem):
        self.persist_boxes()

    def add_box(self, rect: QRectF):
        item = BoundingBoxItem(rect, class_id=0, object_id=None, confidence=None, on_change=self.on_box_change)
        self.scene.addItem(item)
        self.persist_boxes()

    def delete_selected_box(self):
        for item in list(self.scene.selectedItems()):
            if isinstance(item, BoundingBoxItem):
                item_scene = item.scene()
                if item_scene is not None:
                    item_scene.removeItem(item)
        self.persist_boxes()

    def persist_boxes(self):
        if not self.frames: return
        boxes: List[List[float]] = []
        for item in self.scene.items():
            if isinstance(item, BoundingBoxItem):
                rect = item.mapRectToScene(item.rect())
                boxes.append([rect.left(), rect.top(), rect.right(), rect.bottom(), item.class_id, item.confidence])
        self.boxes_per_frame[self.frame_idx] = boxes
        self.box_count_label.setText(f"Boxes: {len(boxes)}")
        self.timeline_view.set_presence(self._build_presence())
        if self.frame_idx in self.saved_state:
            if self._normalize_boxes(boxes) != self._normalize_boxes(self.saved_state[self.frame_idx]):
                del self.saved_state[self.frame_idx]
                self._update_saved_counter()
                self.timeline_view.set_saved_frames(list(self.saved_state.keys()))
                self.update_view()

    def prev_frame(self):
        if self.playing: self.toggle_playback(force_state=False)
        if self.frame_idx > 0:
            self.persist_boxes()
            self.frame_idx -= 1
            self.update_view()

    def next_frame(self):
        if self.playing: self.toggle_playback(force_state=False)
        if self.frame_idx < len(self.frames) - 1:
            self.persist_boxes()
            self.frame_idx += 1
            self.update_view()

    def save_current_frame(self):
        if self.video_path is None or not self.frames: return
        if self.frame_idx in self.saved_state:
            out_dir = self.output_dir / self.video_path.stem
            img_path = out_dir / "images" / f"{self.frame_idx:05d}.jpg"
            lbl_path = out_dir / "labels" / f"{self.frame_idx:05d}.txt"
            if img_path.exists():
                img_path.unlink()
            if lbl_path.exists():
                lbl_path.unlink()
            del self.saved_state[self.frame_idx]
            self._update_saved_counter()
            self.timeline_view.set_saved_frames(list(self.saved_state.keys()))
            self.update_view()
            return

        self.persist_boxes()
        out_dir = self.output_dir / self.video_path.stem
        img_dir = out_dir / "images"
        lbl_dir = out_dir / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        name_base = f"{self.frame_idx:05d}"
        cv2.imwrite(str(img_dir / f"{name_base}.jpg"), self.frames[self.frame_idx])

        h, w, _ = self.frames[self.frame_idx].shape
        with open(lbl_dir / f"{name_base}.txt", "w") as f:
            for x1, y1, x2, y2, _cls, _ in self.boxes_per_frame[self.frame_idx]:
                xc, yc = ((x1 + x2) / 2) / w, ((y1 + y2) / 2) / h
                bw, bh = (x2 - x1) / w, (y2 - y1) / h
                f.write(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")
        
        self.saved_state[self.frame_idx] = list(self.boxes_per_frame[self.frame_idx])
        self.update_view()
        self._update_saved_counter()
        self.timeline_view.set_saved_frames(list(self.saved_state.keys()))

    def next_video(self):
        confirm = QMessageBox.question(
            self,
            "Next Video",
            "Are you sure you want to move to the next video?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if confirm != QMessageBox.Yes:
            return
        self.persist_boxes()
        self.find_and_load_flagged_video()

    def on_timeline_frame_selected(self, frame_idx: int):
        if self.playing: self.toggle_playback(force_state=False)
        self.frame_idx = frame_idx
        self.update_view()

    def _confirm_cross_split(self, video_name: str) -> bool:
        if not self.other_split_root:
            return True
        stem = Path(video_name).stem
        other_data = self.other_split_root / "data" / stem
        other_video = self.other_split_root / "videos" / Path(video_name).name
        if not (other_data.exists() or other_video.exists()):
            return True
        details = []
        if other_data.exists():
            details.append(f"- data: {other_data}")
        if other_video.exists():
            details.append(f"- video: {other_video}")
        message = "\n".join(
            [
                f"'{video_name}' already exists in the other split.",
                *details,
                "",
            "Annotations/test should not share videos.",
                "Proceed anyway?",
            ]
        )
        print(message)
        response = input("Proceed anyway? [y/N]: ").strip().lower()
        return response in ("y", "yes")

    def _build_presence(self) -> dict:
        frames_with_boxes = [i for i, b in enumerate(self.boxes_per_frame) if b]
        return {0: frames_with_boxes}

    @staticmethod
    def _normalize_boxes(boxes: List[List[float]]) -> List[Tuple[float, float, float, float, int]]:
        normalized = []
        for x1, y1, x2, y2, cls, _ in boxes:
            normalized.append((round(x1, 3), round(y1, 3), round(x2, 3), round(y2, 3), int(cls)))
        return sorted(normalized)

    def _update_saved_counter(self):
        self.saved_counter_label.setText(f"Saved: {len(self.saved_state)}")

    def _set_save_button_saved_state(self, saved: bool):
        if saved:
            text_color = QApplication.palette().color(QPalette.HighlightedText).name()
            self.btn_save_frame.setStyleSheet(
                f"QPushButton {{ background-color: #2ecc71; color: {text_color}; font-weight: bold; {self._button_outline_rules()} }}"
            )
        else:
            self.btn_save_frame.setStyleSheet(self._button_outline_style())

    @staticmethod
    def _button_outline_rules() -> str:
        btn_color = QApplication.palette().color(QPalette.Button).name()
        return f"border: 1px solid {btn_color}; border-radius: 4px;"

    @classmethod
    def _button_outline_style(cls) -> str:
        return f"QPushButton {{ {cls._button_outline_rules()} }}"

    def _load_saved_annotations(self):
        if not self.video_path:
            return
        out_dir = self.output_dir / self.video_path.stem
        lbl_dir = out_dir / "labels"
        if not lbl_dir.exists():
            return
        for label_path in sorted(lbl_dir.glob("*.txt")):
            stem = label_path.stem
            frame_idx = None
            if stem.isdigit():
                frame_idx = int(stem)
            elif "_f" in stem:
                idx_str = stem.rsplit("_f", 1)[-1]
                if idx_str.isdigit():
                    frame_idx = int(idx_str)
            if frame_idx is None:
                continue
            if frame_idx < 0 or frame_idx >= len(self.frames):
                continue
            h, w, _ = self.frames[frame_idx].shape
            boxes: List[List[float]] = []
            try:
                lines = label_path.read_text().splitlines()
            except OSError:
                continue
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                try:
                    _cls = int(float(parts[0]))
                    xc, yc, bw, bh = map(float, parts[1:5])
                except ValueError:
                    continue
                x1 = max(0.0, (xc - bw / 2) * w)
                y1 = max(0.0, (yc - bh / 2) * h)
                x2 = min(float(w), (xc + bw / 2) * w)
                y2 = min(float(h), (yc + bh / 2) * h)
                boxes.append([x1, y1, x2, y2, 0, None])
            if boxes:
                self.boxes_per_frame[frame_idx] = boxes
                self.saved_state[frame_idx] = list(boxes)

    def toggle_playback(self, force_state=None):
        self.playing = force_state if force_state is not None else not self.playing
        if self.playing:
            self.play_timer.start(int(1000 / self.fps))
            self.play_button.setText("Pause")
        else:
            self.play_timer.stop()
            self.play_button.setText("Play")

    def advance_playback(self):
        if self.frame_idx < len(self.frames) - 1:
            self.frame_idx += 1
            self.update_view()
        else:
            self.toggle_playback(force_state=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="yolov11s.pt")
    parser.add_argument("--base-dir", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=None)
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument("--video-path", type=str, default=None)
    source_group.add_argument("--box-video", type=str, default=None)
    parser.add_argument("--random-count", type=int, default=5, help="Number of random videos to sample when no video is specified.")
    parser.add_argument("--folder-id", type=str, default=DEFAULT_BOX_FOLDER_ID, help="Box folder ID for random sampling.")
    parser.add_argument("--rebuild-index", action="store_true", help="Rebuild local Box index cache for the folder ID.")
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed for reproducibility.")
    parser.add_argument("--run-new", action="store_true", help="Run inference without preloading saved frames.")
    parser.add_argument("--iou-thresh", type=float, default=0.1)
    parser.add_argument("--flicker-len", type=int, default=3)
    args = parser.parse_args()

    project_root = args.base_dir
    annotations_root = project_root / "annotations"
    test_root = project_root / "test"
    system_files_dir = project_root / "system_files"
    output_dir = args.output_dir or (annotations_root / "data")

    app = QApplication(sys.argv)
    model = YOLO(args.weights)
    navigator = BoxNavigator(
        base_dir=str(annotations_root),
        system_files_dir=str(system_files_dir),
        download_dir=str(annotations_root / "videos"),
    )
    if args.video_path:
        candidates = []
        forced_video = Path(args.video_path)
    elif args.box_video:
        candidates = []
        forced_video = Path(args.box_video)
    else:
        candidates = select_random_box_video_names(
            navigator,
            count=args.random_count,
            folder_id=args.folder_id,
            rebuild_index=args.rebuild_index,
            seed=args.seed,
        )
        forced_video = None
        if not candidates:
            raise SystemExit("No random videos available to download.")

    if output_dir.resolve().is_relative_to(annotations_root.resolve()):
        other_split_root = test_root
    elif output_dir.resolve().is_relative_to(test_root.resolve()):
        other_split_root = annotations_root
    else:
        other_split_root = None

    window = Error_Detector(
        model=model, navigator=navigator, candidates=candidates,
        output_dir=output_dir, iou_threshold=args.iou_thresh,
        flicker_len=args.flicker_len, forced_video=forced_video,
        run_new=args.run_new,
        other_split_root=other_split_root,
    )
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
