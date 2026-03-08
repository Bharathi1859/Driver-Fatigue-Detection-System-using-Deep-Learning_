# 🚗 Driver Fatigue Detection System using Deep Learning

An AI-based system that detects driver drowsiness in real-time using computer vision and deep learning. The system monitors the driver's eyes and yawning behavior using a webcam and alerts the driver if fatigue is detected.

---

# 📌 Project Overview

Driver fatigue is one of the major causes of road accidents worldwide. This project presents a real-time **Driver Fatigue Detection System** that uses deep learning and computer vision to monitor the driver’s facial behavior.

The system analyzes driver images and classifies the driver’s state into multiple categories such as **open eyes, closed eyes, yawning, and normal state**. If fatigue is detected, the system triggers an alert message and alarm sound.

This project also includes a **web dashboard** that allows users to monitor driver fatigue through uploaded images or live webcam input.

---

# 🧠 Technologies Used

• Python
• TensorFlow
• Keras
• OpenCV
• Streamlit
• NumPy
• Matplotlib

---

# 🧠 Deep Learning Model

The fatigue detection system uses a Convolutional Neural Network based on **MobileNetV2**.

The model classifies driver states into the following categories:

* 👁 Open Eyes
* 😴 Closed Eyes
* 🥱 Yawning
* 🙂 No Yawn

Model Accuracy: **~91%**

---

# ⚙️ System Workflow

Driver Face → Image Preprocessing → Deep Learning Model → Classification → Fatigue Detection → Alert System

Steps:

1. Webcam captures driver face
2. Image is preprocessed and resized
3. Deep learning model analyzes facial features
4. System predicts driver state
5. If fatigue detected → alert message + alarm sound

---

# 🖥 Application Dashboard

The project includes a web interface built using Streamlit.

Features include:

• Driver fatigue dashboard
• Upload image detection
• Live webcam monitoring
• Real-time fatigue alerts
• Prediction probability graph

---

# 📸 Project Screenshots

## Dashboard

![Dashboard Screenshot](images/dashboard_1.png)

## Image Upload Detection

![Image Upload](images/dashboard_1.png)

## Live Webcam Detection

![Webcam Detection](images/dashboard_1.png)

---

# 📂 Project Structure

Driver-Fatigue-Detection

```
app.py
model.h5
alarm.wav
requirements.txt
README.md
images/
```

images folder contains project screenshots used in README.

---


# 🚀 Features

✔ Real-time driver fatigue detection
✔ Deep learning based prediction
✔ Eye and yawning detection
✔ Alarm alert system
✔ Interactive web dashboard
✔ Webcam-based monitoring

---

# 📊 Applications

Driver fatigue detection systems can be used in:

• Smart vehicle safety systems
• Driver monitoring systems
• Fleet safety monitoring
• Intelligent transportation systems
• Autonomous vehicles

---

# 🔮 Future Improvements

• Night detection using infrared cameras
• Mobile application integration
• Edge AI deployment
• Advanced driver behavior monitoring

---

# 📚 Author

BHARATHI JAGADEESAN

AI / Machine Learning Project
Driver Fatigue Detection System
Dataset link: https://drive.google.com/drive/folders/1IkLNpad6OBBCaeyZHFRBvhFV8w6uXGCc?usp=drive_link
---

# ⭐ If you like this project

Give it a **star on GitHub ⭐**

