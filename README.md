# 🏋️ OpenCV Squats Counter

This project aims to create a squats counter using the YOLOv3 model for real-time object detection and OpenCV for video processing. The application detects the user's squats and counts them based on the position and movement of the body.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [Support](#support)

## Introduction

The YOLOv3 Squats Counter leverages the YOLOv3 model to detect and count squats performed by a user. The application uses OpenCV to capture video, process frames, and display the results. This project is ideal for those interested in computer vision, deep learning or real-time object detection.

## Features

- 🔍 **Real-time squat detection and counting**
- 🧠 **Uses YOLOv3 for accurate object detection**
- 💻 **Simple and user-friendly interface**

## Installation

### Prerequisites

- 🐍 Python 3.7 or higher
- 💻 Git

1. **Clone the repository:**

    ```bash
    git clone https://github.com/ammarfitwalla/yolov3_squats_counter.git
    cd yolov3_squats_counter
    ```

2. **Download the YOLOv3 models and directory structure:**

    Download the necessary model files and directory structure from the following Google Drive link:

    [Download YOLO Models](https://drive.google.com/drive/folders/1pjyX2TGglypkEijmC6d7zZdnvvt92GhT?usp=sharing)

    After downloading, ensure the following structure is maintained in your project directory:

    ```
    yolo_models/
    ├── face_model/
    │   ├── yolov3-face.cfg
    │   └── yolov3-wider_16000.weights
    └── hand_model/
        ├── yolov3-tiny.cfg
        └── yolov3-tiny_8000.weights
    ```

3. **Install the required Python packages:**

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. **Run the main script:**

    ```bash
    python main.py
    ```

2. **Follow the on-screen instructions:**

    The application will start capturing video from your webcam. Ensure you have enough space to perform squats in front of the camera. The application will detect and count your squats in real-time.

## Project Structure

- `main.py`: The main script to run the squats counter application.
- `yolo_models/`: Directory to store the YOLOv3 model weights and configurations.
  - `face_model/`: Directory for face detection model files.
  - `hand_model/`: Directory for hand detection model files.
- `requirements.txt`: File listing the required Python packages.

## How It Works

1. **Model Initialization:**
   - The `main.py` script initializes the YOLOv3 models for face and hand detection. 
   - Configuration and weights files are loaded from the `yolo_models/` directory.

2. **Video Capture:**
   - OpenCV captures video from the webcam.
   - Frames are processed in real-time to detect faces and hands.

3. **Squat Detection:**
   - The application tracks the position and movement of the user's body.
   - Squats are counted based on specific movement patterns and positions.

4. **Display Results:**
   - The application overlays detection results on the video feed.
   - The squat count is displayed in real-time.

## Troubleshooting

- ⚫ **No video feed / black screen:**
  - Ensure your webcam is properly connected.
  - Check if another application is using the webcam.

- 🗂️ **Model files not found:**
  - Verify that the model files are correctly placed in the `yolo_models/` directory.
  - Ensure the directory structure matches the one described in the installation section.

## Contributing

We welcome contributions to improve this project! Please follow these steps:

1. 🍴 Fork the repository.
2. 🌿 Create a new branch (`git checkout -b feature-branch`).
3. ✨ Make your changes and commit them (`git commit -am 'Add new feature'`).
4. 🔄 Push to the branch (`git push origin feature-branch`).
5. 📥 Create a new Pull Request.

## Support
If you find this project helpful, consider buying me a coffee to support further development:
[☕ Buy Me a Coffee](https://buymeacoffee.com/ammarfitwalla)
