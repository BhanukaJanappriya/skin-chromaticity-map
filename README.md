# 🎨 ChromaSkin Pro: Deterministic Skin Detection Engine

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Stable-brightgreen.svg)]()
[![Methodology](https://img.shields.io/badge/Methodology-Classical%20CV-orange.svg)]()

**ChromaSkin Pro** is a commercial-grade, desktop software for accurate skin region detection using **Chromaticity-based Statistical Modeling**. 

Unlike modern "Black Box" Deep Learning approaches, this tool uses **classical computer vision physics**. It models human skin distribution in the CIE $xy$-chromaticity plane using a Multivariate Gaussian Distribution, ensuring results are **deterministic, explainable, and computationally efficient**.

<img width="1920" height="1025" alt="Screenshot 2026-01-11 152722" src="https://github.com/user-attachments/assets/fdac4bcc-fc58-4e61-966a-4812599111a7" />
<img width="1920" height="1023" alt="Screenshot 2026-01-11 152811" src="https://github.com/user-attachments/assets/038d46a0-7d3e-48c9-bec7-81ead51e0cba" />


---

## 🚀 Key Features

*   **🚫 No Neural Networks:** Pure statistical physics. No GPU required. Runs on standard CPUs.
*   **🖌️ Interactive Teaching:** "Paint" on a reference image to teach the engine specific skin tones (e.g., specific lighting conditions or ethnicities).
*   **📊 Real-time Analytics:** Visualizes your training data in the $xy$ color space with scatter plots.
*   **🎛️ Live Tuning:** Adjust sensitivity sliders to tweak the Mahalanobis distance threshold in real-time.
*   **💾 Model Persistence:** Save trained models (`.cskin` files) and apply them to new images instantly.
*   **📦 Batch Processing:** Automatically process thousands of images in a folder using a saved model.
*   **✨ Professional Export:** Exports results as **Transparent PNGs** (background removed) or Binary Masks.

---

## 🛠️ Installation

### Prerequisites
Ensure you have Python 3.10+ installed.

### 1. Clone the Repository
```bash
git clone https://github.com/BhanukaJanappriya/skin-chromaticity-map.git
cd skin-chromaticity-map
```

---
# 2. Install Dependencies
The project relies on standard scientific computing libraries:
```bash
pip install numpy opencv-python scipy matplotlib PyQt6
```
---

# 📖 Usage Guide
## 1. Launch the Application

```bash
python chromaskin_pro.py
```
## 2. The Workflow
1. Import: Click 📂 Import Image to load a photograph.
2. Teach: Use the mouse to paint green strokes over the skin regions (Face, Neck, Arms).
3. Train: Click ⚡ Train Model. The system fits a Gaussian model to your pixels.
4. Tune: A blue overlay will appear. Use the Sensitivity Slider to adjust the fit.
  * Optional: Toggle "Heatmap Mode" to see probability confidence.
5. Export: Click 💾 Export Image to save the result with the background removed.

## 3. Batch Processing
1. Train a model on one image.
2. Click 📦 Batch Process Folder.
3. Select an Input folder (Source photos) and Output folder.
4. The system will apply your physics model to every image in the folder automatically.

---
## 📐 The Math Behind It

This software relies on **Parametric Statistical Modeling** rather than "black box" Deep Learning. It operates on the principle that human skin tones, regardless of ethnicity, cluster tightly in the **CIE $xy$-Chromaticity Plane** when luminance (brightness) is removed.

### 1. Chromaticity Normalization (Lighting Invariance)
Raw RGB values vary wildly with lighting intensity. We convert pixels to the normalized **CIE $xy$** space to separate **Color (Chromaticity)** from **Brightness (Luminance)**.

Given a pixel with CIE $XYZ$ components:

$$ x = \frac{X}{X + Y + Z}, \quad y = \frac{Y}{X + Y + Z} $$

*Constraint:* We apply a luminance threshold ($X+Y+Z > T$) to filter out dark pixels/shadows where chromaticity becomes unstable (avoiding division-by-zero errors).

### 2. Statistical Learning (Training)
We model the skin distribution as a **Single Multivariate Gaussian**. When the user paints skin samples, we compute the **Maximum Likelihood Estimates (MLE)** for the parameters:

**Mean Vector ($\boldsymbol{\mu}$):** Represents the average skin tone center.

$$ \boldsymbol{\mu} = \frac{1}{N} \sum_{i=1}^{N} \mathbf{x}_i $$

**Covariance Matrix ($\boldsymbol{\Sigma}$):** Represents the shape and orientation of the skin cluster.

$$ \boldsymbol{\Sigma} = \frac{1}{N-1} \sum_{i=1}^{N} (\mathbf{x}_i - \boldsymbol{\mu})(\mathbf{x}_i - \boldsymbol{\mu})^T $$

### 3. Probabilistic Inference (Detection)
For every pixel $\mathbf{x}$ in a new image, we calculate the likelihood that it belongs to the skin model using the **Multivariate Normal Probability Density Function (PDF)**:

$$ P(\mathbf{x}) = \frac{1}{\sqrt{(2\pi)^k |\boldsymbol{\Sigma}|}} \exp\left( -\frac{1}{2} (\mathbf{x} - \boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu}) \right) $$

Where:
*   $k = 2$ (dimensionality of the $xy$ plane).
*   $|\boldsymbol{\Sigma}|$ is the determinant of the covariance matrix.
*   The term in the exponent is related to the squared **Mahalanobis Distance**.

If $P(\mathbf{x}) > \text{Threshold}$, the pixel is classified as skin.

---

# 📂 Project Structure
```bash
chromaskin-pro/
│
├── chromaskin_pro.py    # Single-file complete application source code
├── requirements.txt     # Python dependencies
├── README.md            # Documentation
└── LICENSE              # MIT License```
