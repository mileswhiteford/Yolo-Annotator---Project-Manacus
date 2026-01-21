import sys
from PyQt5.QtWidgets import (
    QGraphicsRectItem,
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QPushButton,
    QLabel,
    QGraphicsView,
    QGraphicsScene,
    QGraphicsPixmapItem,
)
from PyQt5.QtGui import QTransform, QPen, QColor
from PyQt5.QtCore import Qt, QRectF, pyqtSignal, QPointF

class AnnotationView(QGraphicsView):
    new_box_drawn = pyqtSignal(QRectF) 
    navigate_next = pyqtSignal()
    navigate_prev = pyqtSignal()
    delete_selected = pyqtSignal()

    def __init__(self, scene, parent=None):
        super().__init__(scene, parent)
        self.drawing = False
        self.start_point = QPointF()
        self.temp_rect = None 
        self.setFocusPolicy(Qt.StrongFocus)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self._zoom_steps = 0
        self.user_zoomed = False

    def fit_to_pixmap(self, pixmap_item: QGraphicsPixmapItem):
        """Reset zoom to fit the provided pixmap."""
        if pixmap_item and not pixmap_item.pixmap().isNull():
            self.user_zoomed = False
            self._zoom_steps = 0
            self.resetTransform()
            self.fitInView(pixmap_item, Qt.KeepAspectRatio)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            scene_pos = self.mapToScene(event.pos())
            clicked_item = self.scene().itemAt(scene_pos, QTransform())
            # Allow drawing when clicking empty space or the background pixmap
            if clicked_item is None or isinstance(clicked_item, QGraphicsPixmapItem):
                self.drawing = True
                self.start_point = scene_pos

                # Create and add a temporary rectangle for drawing
                self.temp_rect = QGraphicsRectItem(QRectF(self.start_point, self.start_point))
                self.temp_rect.setPen(QPen(QColor(0, 255, 0), 2, Qt.DashLine)) # Green dashed line
                self.scene().addItem(self.temp_rect)
                event.accept()
                return

        super().mousePressEvent(event) # Pass the event to existing items (for moving/selecting)

    def mouseMoveEvent(self, event):
        if self.drawing and self.temp_rect:
            current_point = self.mapToScene(event.pos())
            
            # Calculate the rectangle between start and current points
            new_rect = QRectF(self.start_point, current_point).normalized()
            self.temp_rect.setRect(new_rect)
            event.accept()
            return

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.drawing and self.temp_rect:
            # 1. Finalize the box geometry
            final_rect = self.temp_rect.rect()
            
            # 2. Remove the temporary visualization box
            self.scene().removeItem(self.temp_rect)
            self.temp_rect = None
            self.drawing = False
            
            # 3. Emit a signal to the MainWindow to create the permanent BoundingBoxItem
            if final_rect.width() > 5 and final_rect.height() > 5: # Ignore tiny clicks
                 self.new_box_drawn.emit(final_rect)
            
            event.accept()
            return

        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Right:
            self.navigate_next.emit()
            event.accept()
            return
        if event.key() == Qt.Key_Left:
            self.navigate_prev.emit()
            event.accept()
            return
        if event.key() in (Qt.Key_Delete, Qt.Key_Backspace):
            self.delete_selected.emit()
            event.accept()
            return
        super().keyPressEvent(event)

    def wheelEvent(self, event):
        # Disable wheel-based zoom so scrolling is exclusive to the timeline view
        event.ignore()
