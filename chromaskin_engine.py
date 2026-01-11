import cv2
import numpy as np
import json
from scipy.stats import multivariate_normal

class ChromaSkinEngine:
    def __init__(self):
        self.mean_vector = None
        self.cov_matrix = None
        self.trained = False
        self.min_luminance = 0.05

    def _bgr_to_xy(self, image_bgr):
        img_float = image_bgr.astype(np.float32) / 255.0
        sum_xyz = np.sum(cv2.cvtColor(img_float, cv2.COLOR_BGR2XYZ), axis=2)
        valid_mask = sum_xyz > self.min_luminance

        xy_grid = np.zeros((image_bgr.shape[0], image_bgr.shape[1], 2), dtype=np.float32)
        safe_sum = sum_xyz.copy()
        safe_sum[safe_sum == 0] = 1e-6

        img_xyz = cv2.cvtColor(img_float, cv2.COLOR_BGR2XYZ)
        xy_grid[..., 0] = img_xyz[..., 0] / safe_sum
        xy_grid[..., 1] = img_xyz[..., 1] / safe_sum

        return xy_grid, valid_mask

    def train_model(self, image_bgr, roi_mask):
        xy_grid, valid_mask = self._bgr_to_xy(image_bgr)
        training_mask = (roi_mask > 0) & valid_mask
        training_pixels = xy_grid[training_mask]

        if len(training_pixels) < 50:
            raise ValueError("Not enough valid pixels to train.")

        self.mean_vector = np.mean(training_pixels, axis=0)
        self.cov_matrix = np.cov(training_pixels, rowvar=False)
        self.trained = True

        # Return pixels for the Scatter Plot visualization
        return training_pixels

    def detect_skin(self, image_bgr):
        if not self.trained: return None, None

        h, w = image_bgr.shape[:2]
        xy_grid, valid_mask = self._bgr_to_xy(image_bgr)
        pixels = xy_grid[valid_mask]

        rv = multivariate_normal(self.mean_vector, self.cov_matrix)
        pdf_values = rv.pdf(pixels)

        prob_map = np.zeros((h, w), dtype=np.float32)
        prob_map[valid_mask] = pdf_values

        if np.max(prob_map) > 0:
            prob_map /= np.max(prob_map)

        return prob_map, None # We handle thresholding in UI now

    # --- NEW: PERSISTENCE ---
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
