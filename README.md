# Face Recognition Attendance System

This project is a real-time face recognition system that marks attendance based on identifying known faces from a webcam or video stream. It uses the `face_recognition` library to compare detected faces against a list of known encodings, and logs attendance for recognized individuals.

## Features

- Detects faces in real-time from webcam/video.
- Compares with a database of known face encodings.
- Draws bounding boxes with names on recognized faces.
- Automatically logs attendance with timestamps.
- Scalable to multiple individuals.

## Requirements

- Python 3.7+
- OpenCV
- face_recognition
- NumPy

Install dependencies using pip:

```bash
pip install opencv-python face_recognition numpy
```
