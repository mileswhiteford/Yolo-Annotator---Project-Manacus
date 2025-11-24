from PyQt5.QtWidgets import QGraphicsRectItem, QGraphicsItem
from PyQt5.QtGui import QPen, QBrush, QColor, QCursor
from PyQt5.QtCore import QRectF, Qt, QPointF

class BoundingBoxItem(QGraphicsRectItem):
    """
    A custom QGraphicsRectItem representing a single bounding box annotation.
    It is selectable and movable by the user.
    """
    DEFAULT_CLASS_ID = 0 
    CLASS_ID = DEFAULT_CLASS_ID
    HANDLE_MARGIN = 6
    MIN_SIZE = 4

    def __init__(self, rect: QRectF, class_id: int = DEFAULT_CLASS_ID, parent=None):
        super().__init__(rect, parent)
        
        self.class_id = class_id 
        self.setFlags(QGraphicsItem.ItemIsSelectable | 
                      QGraphicsItem.ItemIsMovable | 
                      QGraphicsItem.ItemSendsGeometryChanges)

        self.setPen(QPen(QColor(255, 0, 0), 2)) 
        self.setBrush(QBrush(Qt.transparent))
        
        self.is_resizing = False
        self.initial_rect = rect 
        self.resize_mode = None
        self.resize_start_pos = QPointF()

        self.setAcceptHoverEvents(True)


    def itemChange(self, change, value):
        """
        Overrides the base method to monitor changes (like movement).
        This is where you could enforce constraints (e.g., box must stay inside video bounds).
        """
        
        if change == QGraphicsItem.ItemSelectedChange:
            if value == True:
                self.setPen(QPen(QColor(255, 255, 0), 3)) # Yellow border when selected
            else:
                self.setPen(QPen(QColor(255, 0, 0), 2))
        
        if change == QGraphicsItem.ItemPositionChange:
            # Value holds the new proposed position (QPointF)
            # could add bounding logic here:
            # e.g., if new_x < 0: return 0
            pass

        return super().itemChange(change, value)

    def hoverMoveEvent(self, event):
        handle = self._handle_at(event.pos())
        self._update_cursor(handle)
        super().hoverMoveEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            handle = self._handle_at(event.pos())
            if handle:
                self.is_resizing = True
                self.resize_mode = handle
                self.initial_rect = QRectF(self.rect())
                self.resize_start_pos = QPointF(event.pos())
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.is_resizing and self.resize_mode:
            new_rect = QRectF(self.initial_rect)
            delta = event.pos() - self.resize_start_pos

            if "left" in self.resize_mode:
                new_left = new_rect.left() + delta.x()
                if new_rect.right() - new_left >= self.MIN_SIZE:
                    new_rect.setLeft(new_left)

            if "right" in self.resize_mode:
                new_right = new_rect.right() + delta.x()
                if new_right - new_rect.left() >= self.MIN_SIZE:
                    new_rect.setRight(new_right)

            if "top" in self.resize_mode:
                new_top = new_rect.top() + delta.y()
                if new_rect.bottom() - new_top >= self.MIN_SIZE:
                    new_rect.setTop(new_top)

            if "bottom" in self.resize_mode:
                new_bottom = new_rect.bottom() + delta.y()
                if new_bottom - new_rect.top() >= self.MIN_SIZE:
                    new_rect.setBottom(new_bottom)

            self.prepareGeometryChange()
            self.setRect(new_rect)
            event.accept()
            return

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.is_resizing:
            self.is_resizing = False
            self.resize_mode = None
            self._update_cursor(None)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def _handle_at(self, pos: QPointF):
        """Return which handle (if any) the point is near."""
        rect = self.rect()
        x, y = pos.x(), pos.y()
        margin = self.HANDLE_MARGIN

        left = abs(x - rect.left()) <= margin
        right = abs(x - rect.right()) <= margin
        top = abs(y - rect.top()) <= margin
        bottom = abs(y - rect.bottom()) <= margin

        if left and top:
            return "top_left"
        if right and top:
            return "top_right"
        if left and bottom:
            return "bottom_left"
        if right and bottom:
            return "bottom_right"
        if top and rect.left() + margin < x < rect.right() - margin:
            return "top"
        if bottom and rect.left() + margin < x < rect.right() - margin:
            return "bottom"
        if left and rect.top() + margin < y < rect.bottom() - margin:
            return "left"
        if right and rect.top() + margin < y < rect.bottom() - margin:
            return "right"
        return None

    def _update_cursor(self, handle):
        cursors = {
            "top_left": Qt.SizeFDiagCursor,
            "bottom_right": Qt.SizeFDiagCursor,
            "top_right": Qt.SizeBDiagCursor,
            "bottom_left": Qt.SizeBDiagCursor,
            "left": Qt.SizeHorCursor,
            "right": Qt.SizeHorCursor,
            "top": Qt.SizeVerCursor,
            "bottom": Qt.SizeVerCursor,
        }
        if handle:
            self.setCursor(QCursor(cursors.get(handle, Qt.ArrowCursor)))
        else:
            self.setCursor(QCursor(Qt.ArrowCursor))
