import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import time
import base64
import winsound
# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="Driver Fatigue Detection",
    page_icon="🚗",
    layout="wide"
)

# ---------------- LOAD MODEL ----------------

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model.h5")

model = load_model()

class_names = ["Closed","Open","no_yawn","yawn"]

# ---------------- SESSION STATE ----------------

if "alarm_triggered" not in st.session_state:
    st.session_state.alarm_triggered = False

# ---------------- PLAY ALARM ----------------

def play_alarm():
    winsound.PlaySound("alarm.wav", winsound.SND_FILENAME | winsound.SND_ASYNC)

# ---------------- SIDEBAR ----------------

st.sidebar.title("🚗 Driver Fatigue Detection")

page = st.sidebar.radio(
    "Navigation",
    ["Dashboard","Upload Image Detection","Live Webcam Detection"]
)

st.sidebar.markdown("---")
st.sidebar.write("Model Accuracy: **91%**")
st.sidebar.write("Model: **MobileNetV2**")

# ---------------- HEADER ----------------

st.title("🚗 Driver Fatigue Detection System")
st.markdown("---")

# ================= DASHBOARD =================

if page == "Dashboard":

    st.subheader("Project Overview")

    st.write("""
    This system detects driver fatigue using deep learning.

    The model classifies driver states into:

    • Closed Eyes  
    • Open Eyes  
    • No Yawn  
    • Yawn
    """)

    c1,c2,c3 = st.columns(3)

    c1.metric("Test Accuracy","91%")
    c2.metric("Classes","4")
    c3.metric("Model","MobileNetV2")

# ================= IMAGE DETECTION =================

elif page == "Upload Image Detection":

    st.subheader("Upload Driver Image")

    uploaded_file = st.file_uploader(
        "Upload Image",
        type=["jpg","png","jpeg"]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file)

        col1,col2 = st.columns(2)

        with col1:
            st.image(image,caption="Uploaded Image",use_column_width=True)

        img = image.resize((224,224))
        img = np.array(img)/255.0
        img = np.expand_dims(img,axis=0)

        prediction = model.predict(img)

        pred_index = np.argmax(prediction)
        predicted_class = class_names[pred_index]

        confidence = prediction[0]
        score = confidence[pred_index]*100

        with col2:

            st.subheader("Prediction Result")

            if predicted_class=="Closed":
                st.error(f"Eyes Closed ⚠️ ({score:.2f}%)")

            elif predicted_class=="yawn":
                st.warning(f"Driver Yawning 😴 ({score:.2f}%)")

            else:
                st.success(f"Driver Alert ✅ ({score:.2f}%)")

            fig,ax = plt.subplots()

            ax.bar(class_names,confidence)

            ax.set_ylabel("Confidence")
            ax.set_title("Prediction Probability")

            st.pyplot(fig)

# ================= WEBCAM DETECTION =================

elif page == "Live Webcam Detection":

    st.subheader("Live Driver Monitoring")

    start = st.button("Start Camera")
    stop = st.button("Stop Camera")

    FRAME_WINDOW = st.image([])
    graph_placeholder = st.empty()

    camera = cv2.VideoCapture(0)

    if start:

        while True:

            ret,frame = camera.read()

            if not ret:
                st.error("Camera error")
                break

            img = cv2.resize(frame,(224,224))
            img = img/255.0
            img = np.expand_dims(img,axis=0)

            prediction = model.predict(img)

            pred_index = np.argmax(prediction)
            predicted_class = class_names[pred_index]

            confidence = prediction[0]

            # Draw label
            cv2.putText(
                frame,
                predicted_class,
                (20,40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,255,0),
                2
            )

            # --------- FATIGUE ALERT ----------

            if predicted_class in ["Closed","yawn"]:

                cv2.putText(
                    frame,
                    "DROWSINESS ALERT!",
                    (20,80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0,0,255),
                    3
                )

                st.error("⚠ DRIVER DROWSINESS DETECTED!")

                if not st.session_state.alarm_triggered:

                    play_alarm()

                    st.session_state.alarm_triggered = True

            else:
                st.session_state.alarm_triggered = False

            # Convert frame
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

            FRAME_WINDOW.image(frame)

            # --------- LIVE PREDICTION GRAPH ----------

            fig,ax = plt.subplots()

            ax.bar(class_names,confidence)

            ax.set_ylim([0,1])

            ax.set_title("Live Prediction Confidence")

            graph_placeholder.pyplot(fig)

            time.sleep(0.15)

            if stop:
                break

        camera.release()