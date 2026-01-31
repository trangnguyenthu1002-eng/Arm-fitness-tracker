import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from streamlit_webrtc import webrtc_streamer, WebRtcMode

# Khởi tạo toàn cục để tránh lỗi load module trong class
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

class BicepsCurlTracker:
    def __init__(self):
        # Thiết lập Pose Landmark
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

    def process(self, image):
        image = cv2.flip(image, 1) # Soi gương cho dễ tập
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # Lấy tọa độ tay phải (hoặc trái tùy bạn chọn)
            shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                     landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                     landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
            
            angle = self.calculate_angle(shoulder, elbow, wrist)
            
            # Logic đếm
            if angle > 160: self.stage = "xuong"
            if angle < 30 and self.stage == 'xuong':
                self.stage = "len"
                self.count += 1
            
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
        return image, self.count, self.stage

# Giao diện Streamlit
st.title("💪 AI Bicep Curl Counter")

# Biến để lưu trạng thái đếm
if "counter" not in st.session_state:
    st.session_state.counter = 0

tracker = BicepsCurlTracker()

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    processed_img, count, stage = tracker.process(img)
    # Cập nhật UI có thể cần logic khác tùy phiên bản webrtc, 
    # nhưng đây là khung chuẩn cho xử lý ảnh.
    return processed_img

webrtc_streamer(
    key="bicep-curl",
    mode=WebRtcMode.SENDRECV,
    video_frame_callback=video_frame_callback,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False},
)