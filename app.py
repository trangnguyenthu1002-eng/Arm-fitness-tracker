import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from streamlit_webrtc import webrtc_streamer, WebRtcMode

# Khởi tạo Mediapipe bên ngoài class để tránh lỗi module
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

class ExerciseTracker:
    def __init__(self):
        self.pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.count = 0
        self.stage = None

    def calculate_angle(self, a, b, c):
        a, b, c = np.array(a), np.array(b), np.array(c)
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians*180.0/np.pi)
        return 360-angle if angle > 180.0 else angle

    def process(self, image, ex_type):
        image = cv2.flip(image, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # Tọa độ các khớp cơ bản
            shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                     landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                     landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
            hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                   landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]

            angle = 0
            # --- LOGIC TỪNG BÀI TẬP ---
            if ex_type == "Bicep Curl":
                angle = self.calculate_angle(shoulder, elbow, wrist)
                if angle > 160: self.stage = "xuong"
                if angle < 30 and self.stage == 'xuong':
                    self.stage, self.count = "len", self.count + 1

            elif ex_type == "Overhead Press":
                angle = self.calculate_angle(shoulder, elbow, wrist)
                if angle < 60: self.stage = "xuong"
                if angle > 160 and self.stage == 'xuong':
                    self.stage, self.count = "len", self.count + 1

            elif ex_type == "Lateral Raise":
                # Góc giữa khuỷu tay - vai - hông
                angle = self.calculate_angle(elbow, shoulder, hip)
                if angle < 30: self.stage = "xuong"
                if angle > 80 and self.stage == 'xuong':
                    self.stage, self.count = "len", self.count + 1

            # Vẽ skeleton và thông tin
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            cv2.rectangle(image, (0,0), (250, 80), (245, 117, 16), -1)
            cv2.putText(image, f'REP: {self.count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(image, f'STATE: {self.stage}', (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
        return image

# --- GIAO DIỆN STREAMLIT ---
st.set_page_config(page_title="AI Fitness Pro", layout="wide")
st.title("🏋️‍♂️ AI Universal Fitness Tracker")

choice = st.sidebar.selectbox("Chọn bài tập:", ["Bicep Curl", "Overhead Press", "Lateral Raise"])
st.sidebar.info(f"Đang tập: {choice}")

if 'tracker' not in st.session_state:
    st.session_state.tracker = ExerciseTracker()

# Nút reset số lần tập
if st.sidebar.button("Reset Counter"):
    st.session_state.tracker.count = 0
    st.session_state.tracker.stage = None

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    processed_img = st.session_state.tracker.process(img, choice)
    return av.VideoFrame.from_ndarray(processed_img, format="bgr24")

import av
webrtc_streamer(
    key="fitness-pro",
    mode=WebRtcMode.SENDRECV,
    video_frame_callback=video_frame_callback,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False},
)