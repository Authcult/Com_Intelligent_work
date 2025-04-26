# canvas.py
from PyQt5.QtWidgets import QWidget, QSizePolicy
from PyQt5.QtGui import QPainter, QPen, QImage, QColor
from PyQt5.QtCore import Qt, QPoint, QSize

class Canvas(QWidget):
    """可绘制的画布控件"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_StaticContents)
        self.setObjectName("CanvasWidget") # 设置 objectName 以便应用样式
        self.drawing = False
        self.last_point = QPoint()
        # 使用白色背景初始化
        self.image = QImage(self.sizeHint(), QImage.Format_RGB32) # 使用 sizeHint 初始化大小
        self.image.fill(Qt.white)
        self.pen_width = 18 # 稍粗的笔触，便于识别
        self.pen_color = Qt.black
        # 设置策略，允许控件缩放
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def set_pen_color(self, color):
        self.pen_color = color

    def set_pen_width(self, width):
        self.pen_width = width

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.last_point = event.pos()

    def mouseMoveEvent(self, event):
        if (event.buttons() & Qt.LeftButton) and self.drawing:
            painter = QPainter(self.image)
            painter.setRenderHint(QPainter.Antialiasing) # 抗锯齿
            painter.setPen(QPen(self.pen_color, self.pen_width, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            painter.drawLine(self.last_point, event.pos())
            self.last_point = event.pos()
            self.update() # 请求重绘

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.drawing:
            self.drawing = False

    def paintEvent(self, event):
        canvas_painter = QPainter(self)
        # 将内部图像绘制到控件上
        canvas_painter.drawImage(self.rect(), self.image, self.image.rect())

    def resizeEvent(self, event):
        if self.width() > self.image.width() or self.height() > self.image.height():
            # 如果窗口变大，创建新的、更大的图像
            new_image = QImage(event.size(), QImage.Format_RGB32)
            new_image.fill(Qt.white)
            painter = QPainter(new_image)
            painter.drawImage(0, 0, self.image) # 将旧图像画到新图像上
            painter.end()
            self.image = new_image
        super().resizeEvent(event)

    def clear_canvas(self):
        """清除画布内容"""
        self.image.fill(Qt.white)
        self.update()

    def get_image(self):
        """获取当前画布内容的 QImage 副本，并将其转换为黑白反转图像"""
        image_copy = self.image.copy()
        return image_copy


    # 推荐的初始大小
    def sizeHint(self):
        return QSize(56, 56)

    # 最小允许的大小
    def minimumSizeHint(self):
        return QSize(48, 48)