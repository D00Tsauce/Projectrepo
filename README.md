# Computer Vision & Deep Learning Based ADAS System

An Advanced Driver Assistance System (ADAS) leveraging the convergence of **Machine Learning** and **Digital Signal Processing (DSP)** for real-time traffic obstacle tracking and automated lane boundary identification. Designed specifically for the Electronics & Communication Engineering (ECE) Domain.

## 👥 Core Project Team
* **N G VINAYAK** (Lead Developer & Integrator)
* **Prabhav P** (ML Engineering & Object Tracking)
* **Pranav PK** (DSP Feature Architecture)

---

## 🛠️ System Architecture & ECE Signal Chain
The application processes continuous video arrays through a dual-channel framework:

1. **DSP Channel (Lane Boundary Extraction):**
   * Grayscale Dimensionality Reduction
   * Gaussian Spatial Convolution Matrix (Low-Pass Noise Filtering)
   * Canny Gradient Mapping (High-Pass Spatial Filtering)
   * Poly-Masked Region of Interest (ROI) Windowing
   * Probabilistic Hough Line Transform Vectorization
2. **Deep Learning Channel (Obstacle Tracking):**
   * Real-time inference using a pre-trained **YOLOv8 Nano** architecture optimized for embedded hardware deployment.
   * Target Isolation: Class filtering tracking Vehicles, Pedestrians, and Traffic Signs with custom classification constraints.

---

## 📁 Repository Structure
* `src/` - Core processing execution packages (`lane_detector.py`, `object_detector.py`)
* `notebooks/` - Prototyping, feature engineering plots, and signal visualization pipeline
* `app.py` - Streamlit web deployment UI engine

---

## 🚀 Deployment Instructions

### Prerequisites
Ensure Python 3.12+ and system path configurations are updated on your operating device.

```bash
# 1. Initialize Virtual Workspace Environment
python -m venv env

# 2. Activate Workspace Execution Thread
# On Windows:
env\Scripts\activate

# 3. Download Structural Requirements Stack
pip install -r requirements.txt

# 4. Initialize Local Streamlit Server Matrix
streamlit run app.py