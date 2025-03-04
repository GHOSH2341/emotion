import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load the trained emotion detection model
MODEL_PATH = "your_model.h5"  # Update with the correct path
model = load_model(MODEL_PATH)

# Define emotion labels (adjust according to your dataset)
emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# Load OpenCV face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Streamlit App UI
st.title("🎭 Real-Time Emotion Detection")

# Video Capture
run = st.checkbox("Start Webcam")
video_placeholder = st.empty()

if run:
    cap = cv2.VideoCapture(0)  # Open webcam
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.warning("⚠️ Unable to access the webcam.")
            break
        
        # Convert frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            face_roi = gray[y:y + h, x:x + w]  # Extract face
            face_resized = cv2.resize(face_roi, (48, 48))  # Resize to match model input
            face_array = img_to_array(face_resized) / 255.0  # Normalize
            face_array = np.expand_dims(face_array, axis=0)  # Expand dims for model input

            # Predict emotion
            prediction = model.predict(face_array)
            emotion = emotion_labels[np.argmax(prediction)]  # Get the highest probability emotion

            # Draw rectangle and label on the face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Show video in Streamlit
        video_placeholder.image(frame, channels="BGR", use_column_width=True)

    cap.release()
else:
    st.write("☝️ Click 'Start Webcam' to begin real-time emotion detection.")

