import sys
import os
import cv2
import numpy as np
import json
from scipy.stats import multivariate_normal

# PyQt6 Imports
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QFileDialog,
                             QSlider, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
                             QMessageBox, QGroupBox, QCheckBox, QTabWidget, QDialog,
                             QProgressBar, QLineEdit)
from PyQt6.QtGui import QImage, QPixmap, QPainter, QColor
from PyQt6.QtCore import Qt, QPoint

# Matplotlib Imports
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# ==========================================
# 1. THE MATH ENGINE (Core Logic)
# ==========================================
class ChromaSkinEngine:
    def __init__(self):
        self.mean_vector = None
        self.cov_matrix = None
        self.trained = False
        self.min_luminance = 0.05

    def _bgr_to_xy(self, image_bgr):
        """Converts BGR to normalized xy chromaticity, filtering dark pixels."""
        img_float = image_bgr.astype(np.float32) / 255.0
        # Convert to XYZ
        img_xyz = cv2.cvtColor(img_float, cv2.COLOR_BGR2XYZ)
        sum_xyz = np.sum(img_xyz, axis=2)

        # Valid mask (luminance check to avoid division by zero)
        valid_mask = sum_xyz > self.min_luminance

        xy_grid = np.zeros((image_bgr.shape[0], image_bgr.shape[1], 2), dtype=np.float32)
        safe_sum = sum_xyz.copy()
        safe_sum[safe_sum == 0] = 1e-6

        # x = X/Sum, y = Y/Sum
        xy_grid[..., 0] = img_xyz[..., 0] / safe_sum
        xy_grid[..., 1] = img_xyz[..., 1] / safe_sum

        return xy_grid, valid_mask

    def train_model(self, image_bgr, roi_mask):
        """Fits a Gaussian to the pixels painted by the user."""
        xy_grid, valid_mask = self._bgr_to_xy(image_bgr)
        # Only use pixels that are painted AND have valid luminance
        training_mask = (roi_mask > 0) & valid_mask
        training_pixels = xy_grid[training_mask]

        if len(training_pixels) < 50:
            raise ValueError("Not enough valid pixels to train. Paint more skin!")

        self.mean_vector = np.mean(training_pixels, axis=0)
        self.cov_matrix = np.cov(training_pixels, rowvar=False)
        self.trained = True

        return training_pixels

    def detect_skin(self, image_bgr):
        """Calculates the probability map for the whole image."""
        if not self.trained: return None, None

        h, w = image_bgr.shape[:2]
        xy_grid, valid_mask = self._bgr_to_xy(image_bgr)
        pixels = xy_grid[valid_mask]

        # Calculate Gaussian PDF
        rv = multivariate_normal(self.mean_vector, self.cov_matrix)
        pdf_values = rv.pdf(pixels)

        prob_map = np.zeros((h, w), dtype=np.float32)
        prob_map[valid_mask] = pdf_values

        if np.max(prob_map) > 0:
            prob_map /= np.max(prob_map)

        return prob_map, None

    def save_profile(self, filepath):
        if not self.trained: raise ValueError("No model to save.")
        data = {
            "mean": self.mean_vector.tolist(),
            "cov": self.cov_matrix.tolist(),
            "version": "1.0"
        }
        with open(filepath, 'w') as f:
            json.dump(data, f)

    def load_profile(self, filepath):
        with open(filepath, 'r') as f:
            data = json.load(f)
        self.mean_vector = np.array(data["mean"])
        self.cov_matrix = np.array(data["cov"])
        self.trained = True

# ==========================================
# 2. BATCH PROCESSOR DIALOG
# ==========================================
class BatchDialog(QDialog):
    def __init__(self, engine, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Batch Processor")
        self.resize(500, 350)
        self.engine = engine
        self.setModal(True)

        layout = QVBoxLayout(self)

        layout.addWidget(QLabel("1. Input Folder (Images):"))
        self.txt_in = QLineEdit()
        btn_in = QPushButton("Browse..."); btn_in.clicked.connect(self.sel_in)
        h1 = QHBoxLayout(); h1.addWidget(self.txt_in); h1.addWidget(btn_in)
        layout.addLayout(h1)

        layout.addWidget(QLabel("2. Output Folder (Result Images):"))
        self.txt_out = QLineEdit()
        btn_out = QPushButton("Browse..."); btn_out.clicked.connect(self.sel_out)
        h2 = QHBoxLayout(); h2.addWidget(self.txt_out); h2.addWidget(btn_out)
        layout.addLayout(h2)

        self.lbl_conf = QLabel("3. Confidence Threshold: 50%")
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(1, 99); self.slider.setValue(50)
        self.slider.valueChanged.connect(lambda v: self.lbl_conf.setText(f"3. Confidence Threshold: {v}%"))
        layout.addWidget(self.lbl_conf); layout.addWidget(self.slider)

        self.check_transparent = QCheckBox("Save as Transparent PNG (Uncheck for B/W Mask)")
        self.check_transparent.setChecked(True)
        layout.addWidget(self.check_transparent)

        self.progress = QProgressBar(); self.progress.setValue(0)
        layout.addWidget(self.progress)

        self.btn_run = QPushButton("🚀 Process All Images")
        self.btn_run.clicked.connect(self.run_batch)
        self.btn_run.setStyleSheet("background-color: #007ACC; color: white; padding: 10px;")
        layout.addWidget(self.btn_run)

    def sel_in(self):
        d = QFileDialog.getExistingDirectory(self, "Select Input Folder")
        if d: self.txt_in.setText(d)

    def sel_out(self):
        d = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if d: self.txt_out.setText(d)

    def run_batch(self):
        in_dir = self.txt_in.text()
        out_dir = self.txt_out.text()

        if not self.engine.trained:
            QMessageBox.critical(self, "Error", "No model loaded! Please train or load a .cskin file first.")
            return

        if not os.path.isdir(in_dir) or not os.path.isdir(out_dir):
            return

        files = [f for f in os.listdir(in_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        total = len(files)
        if total == 0: return

        self.btn_run.setEnabled(False)
        thresh = 1.0 - (self.slider.value() / 100.0)

        for i, filename in enumerate(files):
            self.progress.setValue(int((i / total) * 100))
            QApplication.processEvents()

            img_path = os.path.join(in_dir, filename)
            img = cv2.imread(img_path)
            if img is None: continue

            prob_map, _ = self.engine.detect_skin(img)

            # Create Binary Mask
            binary = (prob_map > thresh).astype(np.uint8) * 255
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k)

            save_path = os.path.join(out_dir, f"result_{filename[:-4]}.png")

            if self.check_transparent.isChecked():
                # Save Transparent PNG (Original Pixels + Alpha Mask)
                b, g, r = cv2.split(img)
                rgba = cv2.merge([b, g, r, binary])
                cv2.imwrite(save_path, rgba)
            else:
                # Save B/W Mask
                cv2.imwrite(save_path, binary)

        self.progress.setValue(100)
        QMessageBox.information(self, "Done", f"Processed {total} images.")
        self.btn_run.setEnabled(True)
        self.close()

# ==========================================
# 3. ANALYTICS PLOT WIDGET
# ==========================================
class AnalyticsPlot(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)
        self.ax.set_title("Chromaticity Distribution (xy)")
        self.ax.grid(True, alpha=0.3)

    def update_plot(self, training_pixels, mean, cov):
        self.ax.clear()
        self.ax.set_xlim(0.1, 0.7)
        self.ax.set_ylim(0.1, 0.6)
        self.ax.set_title("Skin Cluster & Gaussian Model")
        self.ax.set_xlabel("x (Redness)")
        self.ax.set_ylabel("y (Greenness)")
        self.ax.grid(True, alpha=0.3)

        subset = training_pixels[::5]
        self.ax.scatter(subset[:,0], subset[:,1], s=1, c='red', alpha=0.3, label='Samples')
        self.ax.scatter(mean[0], mean[1], c='black', marker='x', s=100, label='Mean')
        self.ax.legend()
        self.draw()

# ==========================================
# 4. IMAGE CANVAS WIDGET (Drawing Tool)
# ==========================================
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

        # Background
        rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        qt_img = QImage(rgb_img.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        self.pixmap_item = QGraphicsPixmapItem(QPixmap.fromImage(qt_img))
        self.scene.addItem(self.pixmap_item)

        # Paint Layer
        self.mask_overlay = QImage(w, h, QImage.Format.Format_ARGB32_Premultiplied)
        self.mask_overlay.fill(Qt.GlobalColor.transparent)
        self.overlay_item = QGraphicsPixmapItem(QPixmap.fromImage(self.mask_overlay))
        self.scene.addItem(self.overlay_item)

        # Result Layer
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

# ==========================================
# 5. MAIN APPLICATION
# ==========================================
class ChromaSkinApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ChromaSkin Pro - Complete Edition")
        self.resize(1400, 900)
        self.engine = ChromaSkinEngine()
        self.cached_prob_map = None

        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        self.tab_editor = QWidget()
        self.setup_editor_tab()
        self.tabs.addTab(self.tab_editor, "🛠️ Editor & Processing")

        self.tab_analytics = QWidget()
        self.setup_analytics_tab()
        self.tabs.addTab(self.tab_analytics, "📊 Model Analytics")

    def setup_editor_tab(self):
        layout = QHBoxLayout(self.tab_editor)

        controls = QGroupBox("Controls")
        controls.setFixedWidth(280)
        vbox = QVBoxLayout(controls)

        # 1. Input
        btn_load = QPushButton("📂 Import Image")
        btn_load.clicked.connect(self.load_image_dialog)

        # 2. Batch
        btn_batch = QPushButton("📦 Batch Process Folder")
        btn_batch.clicked.connect(self.open_batch_dialog)

        # 3. Brush
        self.lbl_brush = QLabel("Brush: 20px")
        slider_brush = QSlider(Qt.Orientation.Horizontal)
        slider_brush.setRange(5, 100); slider_brush.setValue(20)
        slider_brush.valueChanged.connect(lambda v: setattr(self.canvas, 'brush_size', v))

        # 4. Train
        self.btn_train = QPushButton("⚡ Train Model")
        self.btn_train.clicked.connect(self.run_training)
        self.btn_train.setStyleSheet("background-color: #007ACC; color: white; font-weight: bold; padding: 5px;")

        # 5. Presets
        grp_presets = QGroupBox("Model Presets")
        lyt_presets = QVBoxLayout(grp_presets)
        self.btn_save_model = QPushButton("Save Current Model")
        self.btn_save_model.clicked.connect(self.save_model)
        self.btn_save_model.setEnabled(False)
        btn_load_model = QPushButton("Load Model File")
        btn_load_model.clicked.connect(self.load_model)
        lyt_presets.addWidget(self.btn_save_model)
        lyt_presets.addWidget(btn_load_model)

        # 6. Tuning
        grp_tune = QGroupBox("Result Tuning")
        lyt_tune = QVBoxLayout(grp_tune)
        self.slider_thresh = QSlider(Qt.Orientation.Horizontal)
        self.slider_thresh.setRange(1, 99); self.slider_thresh.setValue(50)
        self.slider_thresh.sliderReleased.connect(self.update_result_view)
        self.check_heatmap = QCheckBox("Heatmap Mode")
        self.check_heatmap.toggled.connect(self.update_result_view)
        lyt_tune.addWidget(self.slider_thresh)
        lyt_tune.addWidget(self.check_heatmap)

        # 7. EXPORT SINGLE IMAGE (NEW!)
        self.btn_export = QPushButton("💾 Export Image (Transparent)")
        self.btn_export.clicked.connect(self.export_single_result)
        self.btn_export.setStyleSheet("color: green; font-weight: bold;")
        self.btn_export.setEnabled(False)

        # Add all to sidebar
        vbox.addWidget(btn_load)
        vbox.addWidget(btn_batch)
        vbox.addSpacing(10)
        vbox.addWidget(self.lbl_brush)
        vbox.addWidget(slider_brush)
        vbox.addSpacing(10)
        vbox.addWidget(self.btn_train)
        vbox.addWidget(grp_presets)
        vbox.addWidget(grp_tune)
        vbox.addStretch()
        vbox.addWidget(self.btn_export)

        self.canvas = AnnotationCanvas()
        layout.addWidget(controls)
        layout.addWidget(self.canvas)

    def setup_analytics_tab(self):
        layout = QVBoxLayout(self.tab_analytics)
        self.plotter = AnalyticsPlot(self.tab_analytics, width=5, height=4, dpi=100)
        layout.addWidget(self.plotter)
        lbl_info = QLabel("This chart visualizes the pixels you selected in the xy-Chromaticity plane.")
        lbl_info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(lbl_info)

    # --- ACTIONS ---
    def load_image_dialog(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open", "", "Images (*.jpg *.png *.jpeg)")
        if path:
            img = cv2.imread(path)
            if img is not None:
                self.canvas.load_image(img)
                self.btn_export.setEnabled(False)
                if self.engine.trained:
                    self.run_detection()

    def run_training(self):
        if self.canvas.binary_mask is None: return
        try:
            pixels = self.engine.train_model(self.canvas.cv_image, self.canvas.binary_mask)
            self.plotter.update_plot(pixels, self.engine.mean_vector, self.engine.cov_matrix)
            self.btn_save_model.setEnabled(True)
            self.run_detection()
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def run_detection(self):
        self.cached_prob_map, _ = self.engine.detect_skin(self.canvas.cv_image)
        self.btn_export.setEnabled(True)
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

    def export_single_result(self):
        """Saves the current image with background removed (Transparent PNG)"""
        if self.cached_prob_map is None: return

        path, _ = QFileDialog.getSaveFileName(self, "Save Result", "masked_image.png", "PNG Image (*.png)")
        if not path: return

        # 1. Re-calculate mask based on current slider
        threshold = 1.0 - (self.slider_thresh.value() / 100.0)
        binary = (self.cached_prob_map > threshold).astype(np.uint8) * 255

        # 2. Cleanup
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k)

        # 3. Create Transparent PNG
        b, g, r = cv2.split(self.canvas.cv_image)
        # Combine B, G, R, and Alpha(Binary Mask)
        rgba = cv2.merge([b, g, r, binary])

        cv2.imwrite(path, rgba)
        QMessageBox.information(self, "Success", f"Saved transparent image to:\n{path}")

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

    def open_batch_dialog(self):
        if not self.engine.trained:
            QMessageBox.warning(self, "Warning", "Please Train or Load a Model first.")
            return
        dlg = BatchDialog(self.engine, self)
        dlg.exec()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    if hasattr(Qt.ApplicationAttribute, 'AA_EnableHighDpiScaling'):
        QApplication.setAttribute(Qt.ApplicationAttribute.AA_EnableHighDpiScaling, True)
    window = ChromaSkinApp()
    window.show()
    sys.exit(app.exec())
