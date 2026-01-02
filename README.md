# face-recognition-system
This project is a Face Recognition System developed using Python, OpenCV, and the face_recognition library. It is designed to detect, recognize, and identify human faces in images and real-time video streams with high accuracy. The system leverages computer vision and machine learning techniques to locate faces, extract facial features, and compare them against a stored dataset of known faces. It can be used for applications such as attendance systems, access control, security surveillance, and identity verification.

# Key Features
Real-time face detection using webcam input
Face recognition and identification from stored images
Automatic face encoding and comparison
High accuracy using pre-trained deep learning models
Supports multiple faces in a single frame
Simple and modular Python code structure

# Technologies Used
Python
OpenCV (cv2) – image processing and video capture,
face_recognition – facial feature extraction and matching,
NumPy – numerical computations

# How It Works
The system loads and encodes known faces from a dataset.
A camera or image input is processed frame by frame.
Faces are detected and facial features are extracted.

# Use Cases
Student or staff attendance systems
Security and surveillance applications
Face-based authentication systems
Research and learning in computer vision
This project demonstrates practical knowledge of computer vision, image processing, and Python-based machine learning libraries, and serves as a strong foundation for more advanced AI-driven recognition systems.
Detected faces are compared with stored encodings.
The system identifies and labels recognized faces in real time.

# ⚠️ Python Version Warning

This project relies on the face_recognition library, which depends on dlib.

dlib does not currently provide stable support for Python 3.12, and installation may fail.

✅ Recommended: Python 3.10 or 3.11 for smooth installation and reliable performance.
