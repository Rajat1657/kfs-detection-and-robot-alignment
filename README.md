# KFS Detection and Robot Alignment

## Description
A computer vision system that detects KFS symbols, classifies them as real or fake, and aligns a robot to pick them up for the DD Robocon 2026 Problem Statement

## Features
- Real-time KFS symbol detection using YOLO
- Prespective Wrapping using opencv-python
- Symbol detection for better cropping of the symbol to send it to the tenserflow model
- Classification of kfs using tenserflow model (Real or Fake)
- Position estimation for robot alignment
- Integration with teensy that controls the X drive of the Robot to align the bot to the kfs

## Tech Stack
- Python
- OpenCV
- Ultralytics YOLO
- Tensorflow MobileNet V3
- ROS2
- microros

## Workflow
1. Capture image from camera
2. Detect KFS objects using YOLO
3. Classify detected objects (real or fake)
4. Calculate position of the object
5. Send commands to align robot
6. Pick up the object

## How to Run
1. Install dependencies:
   pip install requirements.txt
2. Run detection script:
   python finalrosalign.py

## Applications
- Autonomous robotics
- Industrial sorting systems
- Object detection and manipulation

## Future Improvements
- Improve detection accuracy
- Add depth estimation
- Optimize for real-time performance
