import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode

# Khởi tạo các giải pháp Mediapipe
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

class BicepsCurlTracker:
    def __init__(self):
        # Khởi tạo Mediapipe Pose
        self.pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.count = 0
        self.stage = None

    def calculate_angle(self, a, b, c):
        a = np.array(a) # Vai
        b = np.array(b) # Khuỷu tay
        c = np.array(c) # Cổ tay

        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians*180.0/np.pi)
        
        if angle > 180.0:
            angle = 360-angle
            
        return angle

    def process(self, image):
        # Chuyển màu sang RGB cho Mediapipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        
        # Nhận diện tư thế
        results = self.pose.process(image_rgb)
        
        # Chuyển lại màu sang BGR để vẽ
        image_rgb.flags.writeable = True
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # Lấy tọa độ Vai, Khuỷu tay, Cổ tay (bên trái ví dụ)
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            
            # Tính góc
            angle = self.calculate_angle(shoulder, elbow, wrist)
            
            # Logic đếm Bicep Curl
            if angle > 160:
                self.stage = "down"
            if angle < 30 and self.stage == 'down':
                self.stage = "up"
                self.count += 1
            
            # Vẽ lên màn hình
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            # Hiển thị số lần tập
            cv2.putText(image, f'Count: {self.count}', (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, f'Stage: {self.stage}', (50, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        return image

class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.tracker = BicepsCurlTracker()

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = self.tracker.process(img)
        return img

# --- UI STREAMLIT ---
st.set_page_config(page_title="AI Fitness Tracker", layout="wide")
st.title("💪 AI Arm Fitness Tracker")
st.write("Hướng dẫn: Đứng ngang camera để hệ thống nhận diện khuỷu tay tốt nhất.")

webrtc_streamer(
    key="fitness-tracker",
    mode=WebRtcMode.SENDRECV,
    video_transformer_factory=VideoProcessor,
    async_processing=True,
)