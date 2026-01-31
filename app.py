import streamlit as st
from streamlit_webrtc import webrtc_streamer
import cv2
import mediapipe as mp

st.set_page_config(page_title="Arm Fitness Trainer", layout="wide")
st.title("💪 AI Arm Fitness Trainer")

# Khởi tạo Mediapipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def video_frame_callback(frame):
    image = frame.to_ndarray(format="bgr24")
    
    # Xử lý AI ở đây
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image)

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Vẽ khung xương
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    return frame.from_ndarray(image, format="bgr24")

# Giao diện chọn bài tập bên cánh trái
with st.sidebar:
    st.header("Chọn bài tập:")
    exercise = st.selectbox("", ["Bicep Curl", "Lateral Raise", "Overhead Press"])
    st.button("View Instructions")

# Mở Camera trên Web
webrtc_streamer(key="fitness", video_frame_callback=video_frame_callback)