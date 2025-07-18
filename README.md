# 🖐️ ASL Sign Language Interpreter

A real-time interpreter for American Sign Language (ASL) using your webcam and machine learning 🧠

This project detects hand signs from live webcam input and interprets them into English letters using a trained model and MediaPipe.

---

## 🔧 Features

- 🖐️ Real-time webcam input
- 🧠 Trained on [Video Call ASL-Signs Computer Vision Project] by User: ASL classification (https://universe.roboflow.com/asl-classification/video-call-asl-signs)
- 📺 Displays predicted letter on-screen
- 🧮 Letter detection every 2 seconds
- 🧹 Word resets after 4 seconds of no hand detected
- 📁 Supports custom datasets for retraining

---

## 📷 Demo

[Optional: Add a video or GIF of the running app here]

---

## 📦 Requirements

- Python 3.8+
- OpenCV
- Mediapipe
- TensorFlow/Keras
- Numpy

---

## 📥 Installation

```bash
git clone https://github.com/yourusername/asl-sign-interpreter.git 
cd asl-sign-interpreter
pip install -r requirements.txt
