import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from PyQt5.QtCore import Qt, QRectF, QPoint, pyqtSignal, QTimer
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QKeySequence, QFont, QFontMetrics, QPalette, QColor, QBrush
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
from ultralytics import YOLO

# Assumed relative imports based on your project structure
try:
    from .bounding_box import BoundingBoxItem
    from .box_downloader import BoxNavigator
    from .random_vid import select_random_box_video_names, DEFAULT_BOX_FOLDER_ID
except ImportError:
    # Fallback if running as a standalone script for testing
    class BoundingBoxItem: pass
    class BoxNavigator: pass
    def select_random_box_video_names(*args, **kwargs): return []
    DEFAULT_BOX_FOLDER_ID = "162855223530"


def sample_frames_with_yolo(video_path: Path, model: YOLO, frame_stride: int) -> tuple[list, list, list]:
    frames: list = []
    boxes_per_frame: list = []
    sample_indices: list = []
    stride = max(1, int(frame_stride))
    frame_idx = 0
    for result in model.track(str(video_path), stream=True, persist=True, verbose=True):
        if frame_idx % stride == 0:
            img = result.orig_img
            frames.append(img.copy())
            sample_indices.append(frame_idx)
            f_boxes: list = []
            if result.boxes is not None and len(result.boxes) > 0:
                xyxy = result.boxes.xyxy.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()
                for i in range(len(xyxy)):
                    x1, y1, x2, y2 = xyxy[i]
                    conf = float(confs[i]) if confs is not None else None
                    f_boxes.append([x1, y1, x2, y2, 0, conf])
            boxes_per_frame.append(f_boxes)
        frame_idx += 1
    return frames, boxes_per_frame, sample_indices


def qpixmap_from_bgr(image: np.ndarray) -> QPixmap:
    h, w, _ = image.shape
    qimg = QImage(image.data, w, h, 3 * w, QImage.Format_BGR888)
    return QPixmap.fromImage(qimg.copy())


class SimpleView(QGraphicsView):
    def __init__(self, scene, on_new_box, parent=None):
        super().__init__(scene, parent)
        self._dragging, self._drag_start, self._rubber_band_item = False, None, None
        self.on_new_box = on_new_box
        self.setRenderHint(QPainter.Antialiasing)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self._user_zoomed = False

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            item = self.itemAt(event.pos())
            if item and not isinstance(item, QGraphicsPixmapItem):
                super().mousePressEvent(event)
                return
            self._dragging, self._drag_start = True, self.mapToScene(event.pos())
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._dragging:
            rect = QRectF(self._drag_start, self.mapToScene(event.pos())).normalized()
            if not self._rubber_band_item:
                self._rubber_band_item = self.scene().addRect(rect, QPen(Qt.cyan, 1))
            else: self._rubber_band_item.setRect(rect)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._dragging:
            self._dragging = False
            rect = QRectF(self._drag_start, self.mapToScene(event.pos())).normalized()
            if rect.width() > 5: self.on_new_box(rect)
            if self._rubber_band_item: 
                self.scene().removeItem(self._rubber_band_item)
                self._rubber_band_item = None
        super().mouseReleaseEvent(event)

    def fit_content(self):
        self.fitInView(self.sceneRect(), Qt.KeepAspectRatio)
        self._user_zoomed = False

    def zoom_in(self): self.scale(1.2, 1.2); self._user_zoomed = True
    def zoom_out(self): self.scale(0.8, 0.8); self._user_zoomed = True


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
        if self.marker_handle_rect and self.marker_handle_rect.contains(scene_pos):
            self.dragging_marker = True
            event.accept()
            return
        if self.marker_bar_rect and self.marker_bar_rect.contains(scene_pos):
            frame = self._frame_from_x(scene_pos.x())
            self.set_current_frame(frame)
            self.frame_selected.emit(frame)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.dragging_marker:
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


class FrameCurator(QMainWindow):
    def __init__(
        self,
        model,
        navigator,
        candidates,
        output_dir,
        frame_stride: int = 30,
        forced_video: Optional[Path] = None,
        run_new: bool = False,
        other_split_root: Optional[Path] = None,
        **kwargs,
    ):
        super().__init__()
        self.model, self.navigator, self.candidates = model, navigator, candidates
        self.output_dir = output_dir
        self.video_path, self.frames, self.boxes_per_frame = None, [], []
        self.sample_indices: List[int] = []
        self.sample_index_map: Dict[int, int] = {}
        self.frame_stride = max(1, int(frame_stride))
        self.frame_idx, self.playing = 0, False
        self.saved_state: Dict[int, list] = {} # Fixed: Initialized before load
        self.forced_video = forced_video
        self.forced_consumed = False
        self.run_new = run_new
        self.other_split_root = other_split_root
        
        self.init_ui()
        
        # Shortcuts
        QShortcut(QKeySequence(Qt.Key_Right), self).activated.connect(self.next_frame)
        QShortcut(QKeySequence(Qt.Key_Left), self).activated.connect(self.prev_frame)
        QShortcut(QKeySequence(Qt.Key_Space), self).activated.connect(self.toggle_playback)
        QShortcut(QKeySequence(Qt.Key_Delete), self).activated.connect(self.delete_selected_box)
        QShortcut(QKeySequence(Qt.Key_Backspace), self).activated.connect(self.delete_selected_box)

        self.current_candidate_idx = 0
        self.find_and_load_flagged_video()

    def init_ui(self):
        self.setWindowTitle("Frame Curator")
        self.resize(1450, 950)
        main_layout = QVBoxLayout()
        
        # --- TOP HEADER ---
        header_widget = QFrame()
        header_widget.setFrameShape(QFrame.StyledPanel)
        header_widget.setAutoFillBackground(True)
        header_widget.setStyleSheet("border-radius: 5px;")
        header_layout = QHBoxLayout(header_widget)
        header_layout.setContentsMargins(8, 6, 8, 6)
        header_layout.setSpacing(12)

        primary_font = QFont("Arial", 15, QFont.Bold)
        normal_font = QFont("Arial", 15)
        info_font = QFont("Arial", 15)

        info_vbox = QVBoxLayout()
        info_vbox.setSpacing(2)
        self.title_label = QLabel("Video: --")
        self.title_label.setFont(info_font)
        self.frame_label = QLabel("Frame: 0 / 0")
        self.frame_label.setFont(info_font)
        info_vbox.addWidget(self.title_label)
        info_vbox.addWidget(self.frame_label)

        saved_vbox = QVBoxLayout()
        saved_vbox.setSpacing(2)
        self.saved_counter_label = QLabel("Saved: 0")
        self.saved_counter_label.setFont(normal_font)
        saved_vbox.addWidget(self.saved_counter_label)

        boxes_vbox = QVBoxLayout()
        boxes_vbox.setSpacing(2)
        self.box_count_label = QLabel("Boxes: 0")
        self.box_count_label.setFont(primary_font)
        boxes_vbox.addWidget(self.box_count_label)

        self.reason_label = QLabel("Stride: --")
        self.reason_label.setFont(primary_font)
        self.reason_label.setAlignment(Qt.AlignCenter)

        btn_vbox = QVBoxLayout()
        btn_vbox.setSpacing(4)
        self.btn_save_frame = QPushButton("SAVE FRAME")
        self.btn_save_frame.setMinimumSize(140, 36)
        self.btn_save_frame.setStyleSheet(self._button_outline_style())
        self.btn_save_frame.clicked.connect(self.save_current_frame)

        btn_next_v = QPushButton("NEXT VIDEO")
        btn_next_v.setMinimumSize(140, 30)
        btn_next_v.setStyleSheet(self._button_outline_style())
        btn_next_v.clicked.connect(self.next_video)
        btn_vbox.addWidget(self.btn_save_frame)
        btn_vbox.addWidget(btn_next_v)

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
        main_layout.addWidget(QLabel("Timeline"))
        main_layout.addWidget(self.timeline_view)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

    def find_and_load_flagged_video(self):
        if self.forced_video and not self.forced_consumed:
            self.forced_consumed = True
            if not self._confirm_cross_split(self.forced_video.name):
                print("Aborted due to annotations/test conflict.")
                raise SystemExit(1)
            target_path = self.forced_video
            if not target_path.exists():
                downloaded = self.navigator.download_vid(target_path.name)
                if downloaded:
                    target_path = Path(downloaded)
                else:
                    QMessageBox.warning(self, "Error", f"Could not find {target_path}")
                    return
            self._load_video(Path(target_path))
            return

        while self.current_candidate_idx < len(self.candidates):
            name = self.candidates[self.current_candidate_idx]
            self.current_candidate_idx += 1
            if not self._confirm_cross_split(name):
                print("Aborted due to annotations/test conflict.")
                raise SystemExit(1)
            path = self.navigator.download_vid(name)
            if not path: continue
            self._load_video(Path(path))
            return
        QMessageBox.information(self, "Done", "No more videos found.")

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
            try:
                item = BoundingBoxItem(
                    rect,
                    class_id=int(cls),
                    object_id=None,
                    confidence=conf,
                    on_change=self.on_box_change,
                )
                self.scene.addItem(item)
            except Exception:
                self.scene.addRect(rect, QPen(Qt.green, 2))

        video_name = self.video_path.name if self.video_path else "--"
        original_idx = self.sample_indices[self.frame_idx] if self.sample_indices else self.frame_idx
        self.title_label.setText(f"Video: {video_name}")
        self.frame_label.setText(f"Frame: {original_idx + 1} (sample {self.frame_idx + 1}/{len(self.frames)})")
        self.box_count_label.setText(f"Boxes: {len(self.boxes_per_frame[self.frame_idx])}")
        self._update_saved_counter()

        self.reason_label.setText(f"Stride: {self.frame_stride}")
        self.reason_label.setStyleSheet("color: #636e72; font-weight: normal;")

        if self.frame_idx in self.saved_state:
            self.btn_save_frame.setText("FRAME SAVED ✓")
            self._set_save_button_saved_state(True)
        else:
            self.btn_save_frame.setText("SAVE FRAME")
            self._set_save_button_saved_state(False)

        if not self.view._user_zoomed: self.view.fit_content()
        self.timeline_view.set_current_frame(self.frame_idx)
        self.timeline_view.set_saved_frames(list(self.saved_state.keys()))

    def on_box_change(self, box: BoundingBoxItem):
        self.persist_boxes()

    def add_box(self, rect):
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
        if not self.frames:
            return
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

    def save_current_frame(self):
        if not self.video_path:
            return
        if self.frame_idx in self.saved_state:
            out = self.output_dir / self.video_path.stem
            frame_number = self.sample_indices[self.frame_idx] if self.sample_indices else self.frame_idx
            name_base = f"{frame_number:05d}"
            img_path = out / "images" / f"{name_base}.jpg"
            lbl_path = out / "labels" / f"{name_base}.txt"
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
        out = self.output_dir / self.video_path.stem
        images_dir = out / "images"
        labels_dir = out / "labels"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        frame_number = self.sample_indices[self.frame_idx] if self.sample_indices else self.frame_idx
        name_base = f"{frame_number:05d}"
        cv2.imwrite(str(images_dir / f"{name_base}.jpg"), self.frames[self.frame_idx])

        h, w, _ = self.frames[self.frame_idx].shape
        with open(labels_dir / f"{name_base}.txt", "w") as f:
            for x1, y1, x2, y2, _cls, _ in self.boxes_per_frame[self.frame_idx]:
                xc, yc = ((x1 + x2) / 2) / w, ((y1 + y2) / 2) / h
                bw, bh = (x2 - x1) / w, (y2 - y1) / h
                f.write(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")
        self.saved_state[self.frame_idx] = list(self.boxes_per_frame[self.frame_idx])
        self.update_view()
        self._update_saved_counter()
        self.timeline_view.set_saved_frames(list(self.saved_state.keys()))

    def toggle_playback(self, force_state=None):
        self.playing = force_state if force_state is not None else not self.playing
        if self.playing:
            self.play_timer.start(33)
            self.play_button.setText("Pause")
        else:
            self.play_timer.stop()
            self.play_button.setText("Play")

    def advance_playback(self):
        if self.frame_idx < len(self.frames) - 1:
            self.frame_idx += 1; self.update_view()
        else: self.toggle_playback(force_state=False)

    def next_frame(self):
        if self.playing: self.toggle_playback(force_state=False)
        if self.frame_idx < len(self.frames) - 1:
            self.persist_boxes()
            self.frame_idx += 1
            self.update_view()

    def prev_frame(self):
        if self.playing: self.toggle_playback(force_state=False)
        if self.frame_idx > 0:
            self.persist_boxes()
            self.frame_idx -= 1
            self.update_view()

    def next_video(self): self.find_and_load_flagged_video()
    def on_timeline_frame_selected(self, f):
        if self.playing: self.toggle_playback(force_state=False)
        self.frame_idx = f
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

    def _load_saved_annotations(self):
        if not self.video_path:
            return
        lbl_dir = self.output_dir / self.video_path.stem / "labels"
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
            sample_idx = self.sample_index_map.get(frame_idx)
            if sample_idx is None or sample_idx < 0 or sample_idx >= len(self.frames):
                continue
            h, w, _ = self.frames[sample_idx].shape
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
                    xc, yc, bw, bh = map(float, parts[1:5])
                except ValueError:
                    continue
                x1 = max(0.0, (xc - bw / 2) * w)
                y1 = max(0.0, (yc - bh / 2) * h)
                x2 = min(float(w), (xc + bw / 2) * w)
                y2 = min(float(h), (yc + bh / 2) * h)
                boxes.append([x1, y1, x2, y2, 0, None])
            if boxes:
                self.boxes_per_frame[sample_idx] = boxes
                self.saved_state[sample_idx] = list(boxes)

    def _load_video(self, video_path: Path):
        self.video_path = video_path
        frames, boxes_per_frame, sample_indices = sample_frames_with_yolo(
            video_path,
            self.model,
            self.frame_stride,
        )
        if not frames:
            QMessageBox.warning(self, "Error", f"No frames sampled from {video_path}")
            return
        self.frames = frames
        self.boxes_per_frame = boxes_per_frame
        self.sample_indices = sample_indices
        self.sample_index_map = {idx: i for i, idx in enumerate(sample_indices)}
        self.frame_idx = 0
        self.saved_state.clear()
        if not self.run_new:
            self._load_saved_annotations()
        self.timeline_view.set_total_frames(len(self.frames))
        self.timeline_view.set_presence(self._build_presence())
        self.timeline_view.set_flagged_frames([])
        self.timeline_view.set_saved_frames(list(self.saved_state.keys()))
        self._update_saved_counter()
        self.update_view()

    def _build_presence(self) -> dict:
        frames_with_boxes = [i for i, b in enumerate(self.boxes_per_frame) if b]
        return {0: frames_with_boxes}

    @staticmethod
    def _normalize_boxes(boxes: List[List[float]]) -> List[tuple]:
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="yolov11s.pt")
    parser.add_argument("--base-dir", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=None)
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument("--video-path", type=str, default=None)
    source_group.add_argument("--box-video", type=str, default=None)
    parser.add_argument("--frame-stride", type=int, default=30, help="Sample every N frames.")
    random_count = 5
    parser.add_argument("--folder-id", type=str, default=DEFAULT_BOX_FOLDER_ID, help="Box folder ID for random sampling.")
    parser.add_argument("--rebuild-index", action="store_true", help="Rebuild local Box index cache for the folder ID.")
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed for reproducibility.")
    parser.add_argument("--run-new", action="store_true", help="Run inference without preloading saved frames.")
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
            count=random_count,
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

    window = FrameCurator(
        model=model,
        navigator=navigator,
        candidates=candidates,
        output_dir=output_dir,
        frame_stride=args.frame_stride,
        forced_video=forced_video,
        run_new=args.run_new,
        other_split_root=other_split_root,
    )
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
