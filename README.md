Got it! Let's update the README to include instructions for creating these specific directories and placing the necessary model files in them.

---

# YOLOv3 Squats Counter

This project aims to create a squats counter using the YOLOv3 model for real-time object detection and OpenCV for video processing. The application detects the user's squats and counts them based on the position and movement of the body.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## Introduction

The YOLOv3 Squats Counter leverages the YOLOv3 model to detect and count squats performed by a user. The application uses OpenCV to capture video, process frames, and display the results. This project is ideal for those interested in computer vision, fitness applications, or real-time object detection.

## Features

- Real-time squat detection and counting
- Uses YOLOv3 for accurate object detection
- Simple and user-friendly interface

## Installation

1. **Clone the repository:**

    ```bash
    git clone https://github.com/ammarfitwalla/yolov3_squats_counter.git
    cd yolov3_squats_counter
    ```

2. **Create the necessary directories:**

    The `main.py` script requires certain directories for storing models and other resources. Ensure the following directories are created:

    ```bash
    mkdir -p yolo_models/face_model
    mkdir -p yolo_models/hand_model
    ```

3. **Download the YOLOv3 model weights and configuration files:**

    Download the required files and place them in the appropriate directories:
    
    - For the face detection model:
        - `yolov3-face.cfg`
        - `yolov3-wider_16000.weights`
    
    Place these files in the `yolo_models/face_model` directory.

    - For the hand detection model:
        - `yolov3-tiny.cfg`
        - `yolov3-tiny_8000.weights`
    
    Place these files in the `yolo_models/hand_model` directory.

4. **Install the required Python packages:**

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

## Contributing

We welcome contributions to improve this project! Please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes and commit them (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Create a new Pull Request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.