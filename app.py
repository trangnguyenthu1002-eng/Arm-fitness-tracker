import cv2
import numpy as np
import mediapipe as mp
import time

class BicepsCurlTracker:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )

        self.count = 0
        self.state = "down"
        self.feedback = ""
        self.last_feedback = ""
        self.up_time = None
        self.min_rep_time = 0.4

        # Ngưỡng góc (Giống Bicep Curl của bạn)
        self.FULL_DOWN = 80    
        self.MID_POINT = 120   
        self.FULL_UP = 160     
        self.WRIST_DRIFT = 0.15 

    def calculate_angle(self, a, b, c):
        a, b, c = np.array(a), np.array(b), np.array(c)
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        if angle > 180: angle = 360 - angle
        return angle

    def process_frame(self, frame):
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = self.pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        form_warning = ""
        angle = 0

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            l_s = [landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER].x, landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER].y]
            l_e = [landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW].x, landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW].y]
            l_w = [landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST].x, landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST].y]
            r_s = [landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].x, landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].y]
            r_e = [landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW].x, landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW].y]
            r_w = [landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST].x, landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST].y]

            angle = (self.calculate_angle(l_s, l_e, l_w) + self.calculate_angle(r_s, r_e, r_w)) / 2
            current_time = time.time()

            # Kiểm tra form cơ bản
            if abs(l_w[0] - l_e[0]) > self.WRIST_DRIFT or abs(r_w[0] - r_e[0]) > self.WRIST_DRIFT:
                form_warning = "Keep wrists over elbows!"
            else:
                if self.state == "down" and angle > self.MID_POINT:
                    self.state = "pressing"
                    self.feedback = "Pushing..."
                elif self.state == "pressing":
                    if angle >= self.FULL_UP:
                        self.state = "up"
                        self.up_time = current_time
                    elif angle < self.FULL_DOWN: self.state = "down"
                elif self.state == "up" and (current_time - self.up_time) >= self.min_rep_time:
                    if angle < self.MID_POINT: self.state = "lowering"
                elif self.state == "lowering" and angle <= self.FULL_DOWN:
                    self.count += 1
                    self.state = "down"
                    self.feedback = f"Rep {self.count}! Good job!"

            self.mp_drawing.draw_landmarks(image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

        # Cập nhật last_feedback
        if self.feedback:
            self.last_feedback = self.feedback
        elif form_warning:
            self.last_feedback = form_warning
        else:
            self.last_feedback = "Ready"

        # GIAO DIỆN (Y hệt ảnh mẫu bạn gửi)
        cv2.putText(image, f'Angle: {int(angle)}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)
        cv2.putText(image, f'Count: {self.count}', (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 100, 0), 2)
        cv2.putText(image, f'State: {self.state}', (10, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 140, 0), 2)
        if form_warning:
            cv2.putText(image, form_warning, (10, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        if self.feedback:
            cv2.putText(image, self.feedback, (10, 215), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        return image, self.count, self.last_feedback
    
    def reset(self):
        """Reset counter và state"""
        self.count = 0
        self.state = "down"
        self.feedback = ""
        self.last_feedback = "Reset"
        self.up_time = None

if __name__ == "__main__":
    tracker = BicepsCurlTracker()
    cap = cv2.VideoCapture(0)
    window_name = "Overhead Press Tracker"
    
    # Khởi tạo cửa sổ trước khi vào vòng lặp
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1)
        output, _, _ = tracker.process_frame(frame)
        
        # Kiểm tra xem cửa sổ có còn tồn tại không trước khi hiển thị
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break
            
        cv2.imshow(window_name, output)

        key = cv2.waitKey(1) & 0xFF
        # Thoát nếu nhấn 'q' hoặc ESC (mã 27)
        if key == ord('q') or key == 27:
            break

    # Đảm bảo camera được tắt và mọi cửa sổ bị xóa hẳn khỏi bộ nhớ
    cap.release()
    cv2.destroyAllWindows()
    print("Program closed successfully.")