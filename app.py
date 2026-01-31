import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import cv2
import mediapipe as mp

# Cấu hình giao diện
st.set_page_config(page_title="Arm Fitness Trainer", layout="wide")
st.title("💪 AI Arm Fitness Trainer")

# Khởi tạo Mediapipe bên ngoài class để tránh lỗi Attribute
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose_tracker = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

class FitnessProcessor(VideoProcessorBase):
    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")
        
        # Xử lý hình ảnh với Mediapipe
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose_tracker.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Vẽ các điểm nối xương
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        return frame.from_ndarray(image, format="bgr24")

# Sidebar chọn bài tập
exercise = st.sidebar.selectbox("Chọn bài tập:", ["Bicep Curl", "Lateral Raise", "Overhead Press"])

# Mở camera web
webrtc_streamer(key="fitness-main", video_processor_factory=FitnessProcessor)