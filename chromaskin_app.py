import sys
import cv2
import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QFileDialog,
                             QSlider, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
                             QMessageBox, QGroupBox, QComboBox, QCheckBox)
from PyQt6.QtGui import QImage, QPixmap, QPainter, QColor, QAction
from PyQt6.QtCore import Qt, QPoint

# Import our logic
try:
    from chromaskin_engine import ChromaSkinEngine
except ImportError:
    sys.exit("Error: chromaskin_engine.py not found.")

class AnnotationCanvas(QGraphicsView):
    """ Same Canvas as before, but with added cleanup method """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.drawing = False
        self.brush_size = 20
        self.brush_color = QColor(0, 255, 0, 100)

        self.cv_image = None
        self.mask_overlay = None
        self.pixmap_item = None
        self.overlay_item = None
        self.result_item = None  # To hold the blue mask/heatmap

        self.binary_mask = None

        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)

    def load_image(self, cv_img):
        self.scene.clear()
        self.cv_image = cv_img
        h, w, c = cv_img.shape
        self.binary_mask = np.zeros((h, w), dtype=np.uint8)

        rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        qt_img = QImage(rgb_img.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        self.pixmap_item = QGraphicsPixmapItem(QPixmap.fromImage(qt_img))
        self.scene.addItem(self.pixmap_item)

        self.mask_overlay = QImage(w, h, QImage.Format.Format_ARGB32_Premultiplied)
        self.mask_overlay.fill(Qt.GlobalColor.transparent)
        self.overlay_item = QGraphicsPixmapItem(QPixmap.fromImage(self.mask_overlay))
        self.scene.addItem(self.overlay_item)

        # Placeholder for result
        self.result_item = QGraphicsPixmapItem()
        self.result_item.setZValue(10) # Ensure it sits on top
        self.scene.addItem(self.result_item)

    def paint_on_overlay(self, scene_pos):
        if self.mask_overlay is None: return
        x, y = int(scene_pos.x()), int(scene_pos.y())

        painter = QPainter(self.mask_overlay)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(self.brush_color)
        painter.drawEllipse(QPoint(x, y), self.brush_size // 2, self.brush_size // 2)
        painter.end()
        self.overlay_item.setPixmap(QPixmap.fromImage(self.mask_overlay))

        # Update Numpy mask
        h, w = self.binary_mask.shape
        # Boundary checks to prevent crash
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(self.binary_mask, (x, y), self.brush_size // 2, 1, -1)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.drawing = True
            self.paint_on_overlay(self.mapToScene(event.pos()))
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.drawing:
            self.paint_on_overlay(self.mapToScene(event.pos()))
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.drawing = False
        super().mouseReleaseEvent(event)


class ChromaSkinApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ChromaSkin Pro - Commercial Edition")
        self.resize(1280, 850)
        self.engine = ChromaSkinEngine()

        # CACHE: Store the raw probability map so we don't re-calculate math on slider change
        self.cached_prob_map = None

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        self.layout = QHBoxLayout(main_widget)

        # --- Sidebar ---
        self.controls = QGroupBox("Control Panel")
        self.controls.setFixedWidth(280)
        self.control_layout = QVBoxLayout(self.controls)

        # 1. Inputs
        self.btn_load = QPushButton("📂 Import Image")
        self.btn_load.clicked.connect(self.load_image_dialog)

        self.lbl_brush = QLabel("Brush Size: 20px")
        self.slider_brush = QSlider(Qt.Orientation.Horizontal)
        self.slider_brush.setRange(5, 100)
        self.slider_brush.setValue(20)
        self.slider_brush.valueChanged.connect(self.update_brush)

        # 2. Process
        self.btn_train = QPushButton("⚡ Train & Detect")
        self.btn_train.setStyleSheet("background-color: #007ACC; color: white; font-weight: bold; padding: 10px;")
        self.btn_train.clicked.connect(self.run_processing)
        self.btn_train.setEnabled(False)

        # 3. Tuning (The new features)
        self.group_tune = QGroupBox("Result Tuning")
        self.layout_tune = QVBoxLayout(self.group_tune)

        self.lbl_thresh = QLabel("Sensitivity: 50%")
        self.slider_thresh = QSlider(Qt.Orientation.Horizontal)
        self.slider_thresh.setRange(1, 99)
        self.slider_thresh.setValue(50) # Equivalent to 0.05 prob
        self.slider_thresh.sliderReleased.connect(self.update_result_view) # Update on release

        self.check_heatmap = QCheckBox("Show Heatmap Mode")
        self.check_heatmap.toggled.connect(self.update_result_view)

        self.layout_tune.addWidget(self.lbl_thresh)
        self.layout_tune.addWidget(self.slider_thresh)
        self.layout_tune.addWidget(self.check_heatmap)
        self.group_tune.setEnabled(False)

        # 4. Export
        self.btn_export = QPushButton("💾 Export Result")
        self.btn_export.clicked.connect(self.export_result)
        self.btn_export.setEnabled(False)

        # Assemble Sidebar
        self.control_layout.addWidget(self.btn_load)
        self.control_layout.addSpacing(10)
        self.control_layout.addWidget(self.lbl_brush)
        self.control_layout.addWidget(self.slider_brush)
        self.control_layout.addSpacing(20)
        self.control_layout.addWidget(self.btn_train)
        self.control_layout.addSpacing(20)
        self.control_layout.addWidget(self.group_tune)
        self.control_layout.addStretch()
        self.control_layout.addWidget(self.btn_export)

        self.canvas = AnnotationCanvas()

        self.layout.addWidget(self.controls)
        self.layout.addWidget(self.canvas)

    def load_image_dialog(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.jpg *.jpeg)")
        if path:
            img = cv2.imread(path)
            if img is not None:
                self.canvas.load_image(img)
                self.btn_train.setEnabled(True)
                self.group_tune.setEnabled(False)
                self.btn_export.setEnabled(False)
                self.cached_prob_map = None
                self.statusBar().showMessage(f"Loaded: {path}")

    def update_brush(self):
        val = self.slider_brush.value()
        self.canvas.brush_size = val
        self.lbl_brush.setText(f"Brush Size: {val}px")

    def run_processing(self):
        if self.canvas.binary_mask is None or np.max(self.canvas.binary_mask) == 0:
            QMessageBox.warning(self, "Warning", "Please paint over skin regions first!")
            return

        self.statusBar().showMessage("Fitting Gaussian Model...")
        QApplication.processEvents()

        try:
            # Train
            self.engine.train_model(self.canvas.cv_image, self.canvas.binary_mask)

            # Detect (Get RAW probability map this time)
            # We use a default threshold, but we store the map for slider usage
            self.cached_prob_map, _ = self.engine.detect_skin(self.canvas.cv_image)

            self.group_tune.setEnabled(True)
            self.btn_export.setEnabled(True)
            self.update_result_view() # Visualize immediately
            self.statusBar().showMessage("Detection Complete.")

        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def update_result_view(self):
        """Updates the visual overlay based on Slider + Checkbox"""
        if self.cached_prob_map is None: return

        # 1. Get Slider Threshold (Invert logic: High Slider = High Sensitivity = Lower Threshold)
        # Slider 1-99.
        # Low Slider (10) -> Strict (Threshold 0.9)
        # High Slider (90) -> Loose (Threshold 0.01)
        slide_val = self.slider_thresh.value()
        self.lbl_thresh.setText(f"Sensitivity: {slide_val}%")

        # Logarithmic mapping often feels better, but linear is fine for now
        # Map 0-100 to 1.0-0.0
        threshold = 1.0 - (slide_val / 100.0)

        h, w = self.cached_prob_map.shape

        # 2. Check View Mode
        if self.check_heatmap.isChecked():
            # -- HEATMAP MODE --
            # Normalize 0-255
            norm_map = (self.cached_prob_map * 255).astype(np.uint8)
            # Apply ColorMap
            heatmap = cv2.applyColorMap(norm_map, cv2.COLORMAP_JET)
            # Make semi-transparent
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2BGRA)
            heatmap[..., 3] = 150 # Alpha

            # Where prob is very low, make fully transparent (cleanup background)
            heatmap[norm_map < 5, 3] = 0

            qt_img = QImage(heatmap.data, w, h, 4 * w, QImage.Format.Format_RGBA8888)
            self.canvas.result_item.setPixmap(QPixmap.fromImage(qt_img))

        else:
            # -- BINARY MASK MODE --
            # Apply Threshold
            binary = (self.cached_prob_map > threshold).astype(np.uint8) * 255

            # Morphological Cleanup (Fast)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

            # Visualization (Blue Overlay)
            buffer = np.zeros((h, w, 4), dtype=np.uint8)
            buffer[binary == 255] = [0, 0, 255, 120] # Blue

            qt_img = QImage(buffer.data, w, h, 4 * w, QImage.Format.Format_RGBA8888)
            self.canvas.result_item.setPixmap(QPixmap.fromImage(qt_img))

    def export_result(self):
        if self.cached_prob_map is None: return

        # Ask user where to save
        path, _ = QFileDialog.getSaveFileName(self, "Export Result", "skin_result.png", "PNG Image (*.png)")
        if not path: return

        # Generate the final high-quality mask based on current slider
        slide_val = self.slider_thresh.value()
        threshold = 1.0 - (slide_val / 100.0)
        binary = (self.cached_prob_map > threshold).astype(np.uint8) * 255

        # Cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        # Create Transparent PNG (Original Image + Alpha Channel from Mask)
        # Get original RGB
        b, g, r = cv2.split(self.canvas.cv_image)
        # Create RGBA
        rgba = cv2.merge([b, g, r, binary])

        cv2.imwrite(path, rgba)
        QMessageBox.information(self, "Success", f"Saved transparent skin image to:\n{path}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    if hasattr(Qt.ApplicationAttribute, 'AA_EnableHighDpiScaling'):
        QApplication.setAttribute(Qt.ApplicationAttribute.AA_EnableHighDpiScaling, True)
    window = ChromaSkinApp()
    window.show()
    sys.exit(app.exec())
