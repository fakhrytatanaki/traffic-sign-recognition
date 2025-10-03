Traffic Sign Detection and Classification
=======================================

Overview
--------
This project performs traffic sign detection and classification from images or a live webcam feed. It combines classical computer vision for contour-based shape detection with a one-vs-rest linear SVM for assigning semantic labels (e.g., stop, right_turn, pedestrian_warn).

Features
--------
- Shape detection via contour matching against canonical templates (triangle, circle, square, octagon)
- Histogram-based filtering utility (optional)
- One-vs-rest SVM image classifier using OpenCV ML
- Webcam demo with real-time bounding boxes and labels

Repository Structure
--------------------
- `main.py`: Live webcam demo, edge/contour extraction, shape detection, SVM labeling, visualization
- `SVMOneVsRest.py`: One-vs-rest SVM training and inference utilities
- `shapes.py`: Canonical shape generation and utilities
- `mltest.py`: Simple training/evaluation script for a binary SVM example
- `dataset/` and `small_dataset/`: Example image data organized by class label directories

Requirements
------------
- Python 3.9+
- OpenCV (`opencv-python`)
- NumPy (`numpy`)
- Matplotlib (`matplotlib`)

Installation
------------
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install opencv-python numpy matplotlib
```

Datasets
--------
Datasets are expected under a directory where each subdirectory represents a class label and contains `.png` images, e.g.:
```text
dataset/
  stop/
    0001.png
    ...
  right_turn/
  pedestrian_warn/
  ...
small_dataset/  # a lighter subset for quick testing
```

Usage
-----
- Run the real-time webcam demo:
```bash
python main.py
```
This will open the default camera, detect contours, match shapes, classify cropped regions with the SVM, and display labels and bounding boxes. Press `q` to quit.

- Train and evaluate a simple binary SVM example (stop vs. not-stop):
```bash
python mltest.py
```

Notes and Tips
--------------
- The webcam demo initializes an SVM one-vs-rest model from `small_dataset` for faster startup. Point `SVMOneVsRestModel.loadDataset(...)` to a larger dataset for improved accuracy.
- Ensure camera access is permitted on your OS. If no camera is available, replace the capture loop with a directory of test images.
- For reproducibility and speed, consider resizing images to `IMAGE_DATA_DIMENSIONS` defined in the code.

License
-------
This code is provided for educational and research purposes. Check dataset licensing separately.

