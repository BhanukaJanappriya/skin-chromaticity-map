import sys
import cv2
import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QFileDialog,
                             QSlider, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
                             QMessageBox, QGroupBox, QCheckBox, QTabWidget)
from PyQt6.QtGui import QImage, QPixmap, QPainter, QColor, QAction
from PyQt6.QtCore import Qt, QPoint

# Matplotlib embedding
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

try:
    from chromaskin_engine import ChromaSkinEngine
except ImportError:
    sys.exit("Error: chromaskin_engine.py not found.")

# --- COMPONENT 1: THE GRAPH WIDGET ---
class AnalyticsPlot(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)
        self.ax.set_title("Chromaticity Distribution (xy)")
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.grid(True, alpha=0.3)

    def update_plot(self, training_pixels, mean, cov):
        self.ax.clear()
        # Settings
        self.ax.set_xlim(0.1, 0.7) # Human skin usually falls here
        self.ax.set_ylim(0.1, 0.6)
        self.ax.set_title("Skin Cluster & Gaussian Model")
        self.ax.grid(True, alpha=0.3)

        # 1. Plot Sample Points (Decimate for speed)
        subset = training_pixels[::5] # Take every 5th pixel
        self.ax.scatter(subset[:,0], subset[:,1], s=1, c='red', alpha=0.3, label='Samples')

        # 2. Plot Mean
        self.ax.scatter(mean[0], mean[1], c='black', marker='x', s=100, label='Mean')

        self.ax.legend()
        self.draw()

# --- COMPONENT 2: THE CANVAS (Same as V2) ---
class AnnotationCanvas(QGraphicsView):
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
        self.result_item = None
        self.binary_mask = None
        self.setRenderHint(QPainter.RenderHint.Antialiasing)

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
        self.result_item = QGraphicsPixmapItem()
        self.result_item.setZValue(10)
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
        h, w = self.binary_mask.shape
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(self.binary_mask, (x, y), self.brush_size // 2, 1, -1)

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton: self.drawing = True; self.paint_on_overlay(self.mapToScene(e.pos()))
        super().mousePressEvent(e)
    def mouseMoveEvent(self, e):
        if self.drawing: self.paint_on_overlay(self.mapToScene(e.pos()))
        super().mouseMoveEvent(e)
    def mouseReleaseEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton: self.drawing = False
        super().mouseReleaseEvent(e)

# --- COMPONENT 3: MAIN APP ---
class ChromaSkinApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ChromaSkin Pro - Phase 3: Analytics")
        self.resize(1400, 900)
        self.engine = ChromaSkinEngine()
        self.cached_prob_map = None

        # Tabs
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        # Tab 1: Editor
        self.tab_editor = QWidget()
        self.setup_editor_tab()
        self.tabs.addTab(self.tab_editor, "🛠️ Editor & Processing")

        # Tab 2: Analytics
        self.tab_analytics = QWidget()
        self.setup_analytics_tab()
        self.tabs.addTab(self.tab_analytics, "📊 Model Analytics")

    def setup_editor_tab(self):
        layout = QHBoxLayout(self.tab_editor)

        # Sidebar
        controls = QGroupBox("Controls")
        controls.setFixedWidth(280)
        vbox = QVBoxLayout(controls)

        btn_load = QPushButton("📂 Import Image")
        btn_load.clicked.connect(self.load_image_dialog)

        self.lbl_brush = QLabel("Brush: 20px")
        slider_brush = QSlider(Qt.Orientation.Horizontal)
        slider_brush.setRange(5, 100); slider_brush.setValue(20)
        slider_brush.valueChanged.connect(lambda v: setattr(self.canvas, 'brush_size', v))

        self.btn_train = QPushButton("⚡ Train Model")
        self.btn_train.clicked.connect(self.run_training)
        self.btn_train.setStyleSheet("background-color: #007ACC; color: white;")

        # Preset Manager
        grp_presets = QGroupBox("Model Presets")
        lyt_presets = QVBoxLayout(grp_presets)
        self.btn_save_model = QPushButton("Save Current Model")
        self.btn_save_model.clicked.connect(self.save_model)
        self.btn_save_model.setEnabled(False)

        btn_load_model = QPushButton("Load Model File")
        btn_load_model.clicked.connect(self.load_model)

        lyt_presets.addWidget(self.btn_save_model)
        lyt_presets.addWidget(btn_load_model)

        # Tuning
        grp_tune = QGroupBox("Sensitivity")
        lyt_tune = QVBoxLayout(grp_tune)
        self.slider_thresh = QSlider(Qt.Orientation.Horizontal)
        self.slider_thresh.setRange(1, 99); self.slider_thresh.setValue(50)
        self.slider_thresh.sliderReleased.connect(self.update_result_view)
        self.check_heatmap = QCheckBox("Heatmap Mode")
        self.check_heatmap.toggled.connect(self.update_result_view)
        lyt_tune.addWidget(self.slider_thresh)
        lyt_tune.addWidget(self.check_heatmap)

        vbox.addWidget(btn_load)
        vbox.addWidget(self.lbl_brush)
        vbox.addWidget(slider_brush)
        vbox.addSpacing(10)
        vbox.addWidget(self.btn_train)
        vbox.addWidget(grp_presets)
        vbox.addWidget(grp_tune)
        vbox.addStretch()

        self.canvas = AnnotationCanvas()
        layout.addWidget(controls)
        layout.addWidget(self.canvas)

    def setup_analytics_tab(self):
        layout = QVBoxLayout(self.tab_analytics)
        self.plotter = AnalyticsPlot(self.tab_analytics, width=5, height=4, dpi=100)
        layout.addWidget(self.plotter)

        lbl_info = QLabel("This chart visualizes the pixels you selected in the xy-Chromaticity plane.\nRed Dots = Your pixels. X = The calculated average color.")
        lbl_info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(lbl_info)

    # --- ACTIONS ---
    def load_image_dialog(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open", "", "Images (*.jpg *.png)")
        if path:
            img = cv2.imread(path)
            if img is not None:
                self.canvas.load_image(img)
                # If we have a model loaded, run detection immediately
                if self.engine.trained:
                    self.run_detection()

    def run_training(self):
        if self.canvas.binary_mask is None: return
        try:
            # 1. Train
            pixels = self.engine.train_model(self.canvas.cv_image, self.canvas.binary_mask)
            # 2. Update Plot
            self.plotter.update_plot(pixels, self.engine.mean_vector, self.engine.cov_matrix)
            self.btn_save_model.setEnabled(True)
            # 3. Detect
            self.run_detection()
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def run_detection(self):
        self.cached_prob_map, _ = self.engine.detect_skin(self.canvas.cv_image)
        self.update_result_view()

    def update_result_view(self):
        if self.cached_prob_map is None: return
        threshold = 1.0 - (self.slider_thresh.value() / 100.0)
        h, w = self.cached_prob_map.shape

        if self.check_heatmap.isChecked():
            norm = (self.cached_prob_map * 255).astype(np.uint8)
            vis = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
            vis = cv2.cvtColor(vis, cv2.COLOR_BGR2BGRA); vis[..., 3] = 150
            vis[norm < 5, 3] = 0
        else:
            binary = (self.cached_prob_map > threshold).astype(np.uint8) * 255
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k)
            vis = np.zeros((h, w, 4), dtype=np.uint8)
            vis[binary == 255] = [0, 0, 255, 120]

        qt_img = QImage(vis.data, w, h, 4*w, QImage.Format.Format_RGBA8888)
        self.canvas.result_item.setPixmap(QPixmap.fromImage(qt_img))

    def save_model(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save Skin Model", "skin_model.cskin", "ChromaSkin Model (*.cskin)")
        if path:
            self.engine.save_profile(path)
            QMessageBox.information(self, "Saved", f"Model saved to {path}")

    def load_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Skin Model", "", "ChromaSkin Model (*.cskin)")
        if path:
            self.engine.load_profile(path)
            QMessageBox.information(self, "Loaded", "Model loaded. Applying to current image...")
            if self.canvas.cv_image is not None:
                self.run_detection()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    if hasattr(Qt.ApplicationAttribute, 'AA_EnableHighDpiScaling'):
        QApplication.setAttribute(Qt.ApplicationAttribute.AA_EnableHighDpiScaling, True)
    window = ChromaSkinApp()
    window.show()
    sys.exit(app.exec())
