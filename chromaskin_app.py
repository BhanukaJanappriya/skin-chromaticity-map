import sys
import cv2
import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QFileDialog,
                             QSlider, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
                             QMessageBox, QGroupBox)
from PyQt6.QtGui import QImage, QPixmap, QPainter, QColor, QPen, QAction, QIcon
from PyQt6.QtCore import Qt, pyqtSignal, QPoint

# Import our logic from Step 1
try:
    from chromaskin_engine import ChromaSkinEngine
except ImportError:
    print("CRITICAL: 'chromaskin_engine.py' not found. Please save the Step 1 code first.")
    sys.exit(1)

# ==========================================
#  CUSTOM WIDGET: The Paintable Canvas
# ==========================================
class AnnotationCanvas(QGraphicsView):
    """
    A professional image viewer that supports:
    1. Zooming/Panning (Standard QGraphicsView features)
    2. Painting Mode (Drawing on a transparent overlay)
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)

        # State
        self.drawing = False
        self.brush_size = 20
        self.brush_color = QColor(0, 255, 0, 100) # Semi-transparent Green

        # Data Layers
        self.cv_image = None       # The raw BGR numpy array
        self.mask_overlay = None   # The QImage we paint on
        self.pixmap_item = None    # The base photo item
        self.overlay_item = None   # The painted mask item

        # Internal Masks
        self.binary_mask = None    # Numpy mask (0 or 1) for the Engine

        # Setup Interaction
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setDragMode(QGraphicsView.DragMode.NoDrag) # Default to paint mode

    def load_image(self, cv_img):
        """Reset scene and load new image"""
        self.scene.clear()
        self.cv_image = cv_img
        h, w, c = cv_img.shape

        # 1. Init Binary Mask (Numpy)
        self.binary_mask = np.zeros((h, w), dtype=np.uint8)

        # 2. Create Background Item (The Photo)
        # Convert BGR to RGB for Qt
        rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        qt_img = QImage(rgb_img.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        self.pixmap_item = QGraphicsPixmapItem(QPixmap.fromImage(qt_img))
        self.scene.addItem(self.pixmap_item)

        # 3. Create Overlay Item (The Paint Layer)
        # ARGB32 Premultiplied is best for transparency
        self.mask_overlay = QImage(w, h, QImage.Format.Format_ARGB32_Premultiplied)
        self.mask_overlay.fill(Qt.GlobalColor.transparent)

        self.overlay_item = QGraphicsPixmapItem(QPixmap.fromImage(self.mask_overlay))
        self.scene.addItem(self.overlay_item)

    def paint_on_overlay(self, scene_pos):
        """Draws on the overlay QImage and updates the scene"""
        if self.mask_overlay is None:
            return

        x, y = int(scene_pos.x()), int(scene_pos.y())

        # 1. Update Visuals (Qt Painting)
        painter = QPainter(self.mask_overlay)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(self.brush_color)
        painter.drawEllipse(QPoint(x, y), self.brush_size // 2, self.brush_size // 2)
        painter.end()

        # Refresh the graphics item
        self.overlay_item.setPixmap(QPixmap.fromImage(self.mask_overlay))

        # 2. Update Logic (Numpy Mask)
        # Draw a white circle on the binary mask for the Engine to read
        # Note: In OpenCV, (x, y) order is (col, row)
        cv2.circle(self.binary_mask, (x, y), self.brush_size // 2, 1, -1)

    # --- Mouse Events for Painting ---
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.drawing = True
            # Map window coordinates to Scene (Image) coordinates
            scene_pos = self.mapToScene(event.pos())
            self.paint_on_overlay(scene_pos)
        else:
            # Allow panning with Middle/Right click if needed
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.drawing:
            scene_pos = self.mapToScene(event.pos())
            self.paint_on_overlay(scene_pos)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.drawing = False
        super().mouseReleaseEvent(event)


# ==========================================
#  MAIN WINDOW APPLICATION
# ==========================================
class ChromaSkinApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ChromaSkin Pro - Commercial Architect Edition")
        self.resize(1200, 800)

        # Initialize the Math Engine
        self.engine = ChromaSkinEngine()

        # Main Layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        self.layout = QHBoxLayout(main_widget)

        # --- Left Panel: Controls ---
        self.controls = QGroupBox("Workflow")
        self.controls.setFixedWidth(250)
        self.control_layout = QVBoxLayout(self.controls)

        # 1. Load Button
        self.btn_load = QPushButton("1. Import Image")
        self.btn_load.clicked.connect(self.load_image_dialog)
        self.btn_load.setMinimumHeight(40)

        # 2. Brush Controls
        self.lbl_brush = QLabel("Brush Size: 20px")
        self.slider_brush = QSlider(Qt.Orientation.Horizontal)
        self.slider_brush.setRange(5, 100)
        self.slider_brush.setValue(20)
        self.slider_brush.valueChanged.connect(self.update_brush)

        # 3. Action Button
        self.btn_train = QPushButton("2. Train & Detect")
        self.btn_train.clicked.connect(self.run_processing)
        self.btn_train.setMinimumHeight(50)
        self.btn_train.setStyleSheet("background-color: #007ACC; color: white; font-weight: bold;")
        self.btn_train.setEnabled(False)

        # 4. Export (Stub)
        self.btn_export = QPushButton("3. Export Mask")
        self.btn_export.setEnabled(False)

        # Add to sidebar
        self.control_layout.addWidget(self.btn_load)
        self.control_layout.addSpacing(20)
        self.control_layout.addWidget(self.lbl_brush)
        self.control_layout.addWidget(self.slider_brush)
        self.control_layout.addSpacing(20)
        self.control_layout.addWidget(self.btn_train)
        self.control_layout.addWidget(self.btn_export)
        self.control_layout.addStretch()

        # --- Right Panel: Canvas ---
        self.canvas = AnnotationCanvas()

        # Add to Main Layout
        self.layout.addWidget(self.controls)
        self.layout.addWidget(self.canvas)

    def load_image_dialog(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if path:
            img = cv2.imread(path)
            if img is not None:
                self.canvas.load_image(img)
                self.btn_train.setEnabled(True)
                self.statusBar().showMessage(f"Loaded: {path} | Resolution: {img.shape[1]}x{img.shape[0]}")

    def update_brush(self):
        val = self.slider_brush.value()
        self.canvas.brush_size = val
        self.lbl_brush.setText(f"Brush Size: {val}px")

    def run_processing(self):
        """The Bridge between UI and Engine"""
        if self.canvas.binary_mask is None or np.max(self.canvas.binary_mask) == 0:
            QMessageBox.warning(self, "No Data", "Please paint over some skin regions first!")
            return

        self.statusBar().showMessage("Computing Chromaticity Model... Please wait.")
        QApplication.processEvents() # Force UI update

        try:
            # 1. Train
            self.engine.train_model(self.canvas.cv_image, self.canvas.binary_mask)

            # 2. Detect
            prob_map, result_mask = self.engine.detect_skin(self.canvas.cv_image)

            # 3. Visualize Result
            self.show_results(result_mask)
            self.statusBar().showMessage("Detection Complete. Model Fitted.")
            self.btn_export.setEnabled(True)

        except Exception as e:
            QMessageBox.critical(self, "Processing Error", str(e))

    def show_results(self, result_mask):
        """
        Overlay the Result Mask (Blue) onto the existing view
        """
        h, w = result_mask.shape

        # Create a Blue visualization overlay
        # 0 where mask is 0, Blue where mask is 255

        # Create QImage for result
        res_overlay = QImage(w, h, QImage.Format.Format_ARGB32_Premultiplied)
        res_overlay.fill(Qt.GlobalColor.transparent)

        painter = QPainter(res_overlay)
        # Use a Blue composition mode
        painter.setPen(Qt.PenStyle.NoPen)
        color = QColor(0, 0, 255, 120) # Semi-transparent Blue

        # We need to paint only where result_mask == 255
        # Converting numpy mask to QBitmap or painting pixels is one way,
        # but for speed in Python, let's just create a QImage from buffer

        # Fast Numpy -> QImage ARGB conversion for visualization
        # Create an RGBA buffer
        buffer = np.zeros((h, w, 4), dtype=np.uint8)
        buffer[result_mask == 255] = [0, 0, 255, 120] # Blue, Alpha 120

        result_qimg = QImage(buffer.data, w, h, 4 * w, QImage.Format.Format_RGBA8888)

        # Add result item to scene
        res_item = QGraphicsPixmapItem(QPixmap.fromImage(result_qimg))
        self.canvas.scene.addItem(res_item)


if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Optional: Enable High DPI scaling for modern monitors
    if hasattr(Qt.ApplicationAttribute, 'AA_EnableHighDpiScaling'):
        QApplication.setAttribute(Qt.ApplicationAttribute.AA_EnableHighDpiScaling, True)

    window = ChromaSkinApp()
    window.show()
    sys.exit(app.exec())
