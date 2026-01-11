import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

class ChromaSkinEngine:
    """
    Commercial-grade implementation of Gaussian Chromaticity Skin Detection.
    Strictly deterministic. No Neural Networks.
    """

    def __init__(self):
        self.mean_vector = None
        self.cov_matrix = None
        self.trained = False
        # Constraint: Ignore very dark pixels to avoid chromaticity noise (division by zero)
        self.min_luminance = 0.05

    def _bgr_to_xy_chromaticity(self, image_bgr):
        """
        Converts BGR image to normalized xy chromaticity space.
        Returns:
            xy_grid: (H, W, 2) array of x, y values
            valid_mask: (H, W) boolean array indicating pixels with sufficient luminance
        """
        # 1. Normalize and Float Conversion
        img_float = image_bgr.astype(np.float32) / 255.0

        # 2. Convert to CIE XYZ (Standard observer D65)
        img_xyz = cv2.cvtColor(img_float, cv2.COLOR_BGR2XYZ)

        # 3. Calculate Luminance Sum (X + Y + Z)
        sum_xyz = np.sum(img_xyz, axis=2)

        # 4. Create Validity Mask (Filter out black/dark noise)
        valid_mask = sum_xyz > self.min_luminance

        # 5. Compute xy
        # Initialize canvas
        xy_grid = np.zeros((image_bgr.shape[0], image_bgr.shape[1], 2), dtype=np.float32)

        # Avoid division by zero by adding epsilon where sum is 0 (though masked out later)
        safe_sum = sum_xyz.copy()
        safe_sum[safe_sum == 0] = 1e-6

        # x = X / (X+Y+Z)
        xy_grid[..., 0] = img_xyz[..., 0] / safe_sum
        # y = Y / (X+Y+Z)
        xy_grid[..., 1] = img_xyz[..., 1] / safe_sum

        return xy_grid, valid_mask

    def train_model(self, image_bgr, roi_mask):
        """
        Statistical Learning Step.
        image_bgr: The full image
        roi_mask: Binary mask (same size as image) where 1 = skin sample
        """
        print(">> Processing image for training...")
        xy_grid, valid_mask = self._bgr_to_xy_chromaticity(image_bgr)

        # Combine ROI mask with Luminance validity mask
        # We only learn from pixels that are both USER_SELECTED and VALID_LUMINANCE
        training_mask = (roi_mask > 0) & valid_mask

        # Extract pixel values
        training_pixels = xy_grid[training_mask]

        if len(training_pixels) < 50:
            raise ValueError("Not enough valid pixels selected to train model.")

        # Compute Statistics (Maximum Likelihood Estimation)
        self.mean_vector = np.mean(training_pixels, axis=0)
        self.cov_matrix = np.cov(training_pixels, rowvar=False)

        self.trained = True
        print(f"   [Model Learned] Mean: {self.mean_vector}, Pixels Used: {len(training_pixels)}")
        return training_pixels # Return for visualization purposes

    def detect_skin(self, image_bgr, probability_threshold=0.05):
        """
        Inference Step.
        Returns:
            probability_map: 0.0-1.0 float map
            binary_mask: 0-255 uint8 mask
        """
        if not self.trained:
            raise RuntimeError("Model not trained yet.")

        h, w = image_bgr.shape[:2]
        xy_grid, valid_mask = self._bgr_to_xy_chromaticity(image_bgr)

        # Flatten valid pixels for PDF calculation
        pixels_to_test = xy_grid[valid_mask]

        # 1. Calculate Gaussian Probability Density Function (PDF)
        # Using SciPy for numerical stability
        rv = multivariate_normal(self.mean_vector, self.cov_matrix)
        pdf_values = rv.pdf(pixels_to_test)

        # 2. Reconstruct Image
        probability_map = np.zeros((h, w), dtype=np.float32)
        probability_map[valid_mask] = pdf_values

        # Normalize probability map for display (0.0 to 1.0)
        max_val = np.max(probability_map)
        if max_val > 0:
            probability_map = probability_map / max_val

        # 3. Thresholding
        binary_mask = (probability_map > probability_threshold).astype(np.uint8) * 255

        # 4. Post-Processing (Morphological Cleanup)
        # Explainable AI: Remove salt-noise (erode) then fill holes (dilate)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

        return probability_map, binary_mask

# ==========================================
#  CLI / PROTOTYPE RUNNER
# ==========================================
if __name__ == "__main__":
    # Load an image (Change filename to a local file for testing)
    # Using a generated image if none exists, or try to load 'face.jpg'
    img_path = 'couple.jpg'
    img = cv2.imread(img_path)

    if img is None:
        print(f"Error: Could not load {img_path}. Please place a JPG named 'couple.jpg' in this folder.")
        print("Creating a dummy image for demonstration...")
        img = np.zeros((400, 400, 3), dtype=np.uint8)
        cv2.rectangle(img, (100, 100), (300, 300), (180, 190, 230), -1) # Draw a "skin" colored box
        cv2.rectangle(img, (50, 50), (100, 100), (0, 255, 0), -1) # Draw a green "non-skin" box

    # 1. Interactive Selection
    print("\nINSTRUCTIONS:")
    print("1. A window will pop up.")
    print("2. Draw a box around a SKIN REGION (e.g., forehead/cheek).")
    print("3. Press ENTER or SPACE to confirm.")

    r = cv2.selectROI("Select Skin Sample", img, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow("Select Skin Sample")

    if r[2] == 0 or r[3] == 0:
        print("No selection made. Exiting.")
        exit()

    # Create mask from selection
    roi_mask = np.zeros(img.shape[:2], dtype=np.uint8)
    roi_mask[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])] = 1

    # 2. Instantiate Engine & Train
    engine = ChromaSkinEngine()

    # We get back training pixels just to plot them later
    training_pixels = engine.train_model(img, roi_mask)

    # 3. Run Detection on the whole image
    prob_map, skin_mask = engine.detect_skin(img)

    # 4. Visualization (Matplotlib for Scientific Plotting)
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('ChromaSkin Pro - Statistical Model Analysis', fontsize=16)

    # Original Image
    axs[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axs[0, 0].set_title("Original Image (ROI Boxed)")
    # Draw the box on the plot for reference
    import matplotlib.patches as patches
    rect = patches.Rectangle((r[0], r[1]), r[2], r[3], linewidth=2, edgecolor='r', facecolor='none')
    axs[0, 0].add_patch(rect)

    # Probability Heatmap
    im = axs[0, 1].imshow(prob_map, cmap='jet')
    axs[0, 1].set_title("Probability Heatmap (Gaussian Likelihood)")
    plt.colorbar(im, ax=axs[0, 1])

    # Binary Mask
    axs[1, 0].imshow(skin_mask, cmap='gray')
    axs[1, 0].set_title("Final Segmented Mask")

    # Chromaticity Space (The Physics)
    # Plot a random subset of training pixels to save rendering time
    subset_idx = np.random.choice(len(training_pixels), min(len(training_pixels), 2000), replace=False)
    subset_pixels = training_pixels[subset_idx]

    axs[1, 1].scatter(subset_pixels[:, 0], subset_pixels[:, 1], alpha=0.3, s=1, c='red', label='Skin Samples')
    axs[1, 1].set_xlim(0, 1)
    axs[1, 1].set_ylim(0, 1)
    axs[1, 1].set_xlabel('Normalized x')
    axs[1, 1].set_ylabel('Normalized y')
    axs[1, 1].set_title("Chromaticity Distribution (xy plane)")
    axs[1, 1].legend()
    axs[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print(">> Processing Complete.")
