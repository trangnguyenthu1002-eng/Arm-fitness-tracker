import cv2
import numpy as np
import mediapipe as mp
import time
import pygame
import gc 

class OverheadPressTracker:
    def __init__(self):
        # Audio paths
        self.success_sound_path = "audio/perfect.wav"
        self.background_sound_path = "audio/background_music.mp3"
        self.too_high_sound_path = "audio/too_high.mp3"
        self.bad_form_sound_path = "audio/bad_form.mp3"
        self.try_again_sound_path = "audio/try_again.wav"
        
        # Audio initialization flags
        self.pygame_initialized = False
        self.sounds_loaded = False
        self.sound_enabled = False
        self.music_loaded = False
        self.music_playing = False
        
        # Sound cooldown tracking
        self.last_sound_time = {}  
        self.sound_cooldown = 1.5
        
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        
        self.drawing_spec_good = mp.solutions.drawing_utils.DrawingSpec(
            color=(0, 255, 0), thickness=2, circle_radius=2
        )
        self.drawing_spec_warning = mp.solutions.drawing_utils.DrawingSpec(
            color=(0, 165, 255), thickness=2, circle_radius=2
        )
        self.drawing_spec_error = mp.solutions.drawing_utils.DrawingSpec(
            color=(0, 0, 255), thickness=2, circle_radius=2
        )
        
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
        self.reset()
        
        # Thresholds 
        self.FULL_DOWN = 95      
        self.MID_POINT = 115   
        self.FULL_UP = 150    
        self.WRIST_DRIFT = 0.25  
    
    def reset(self):
        self.count = 0
        self.state = "down"
        self.up_time = None
        self.min_rep_time = 0.5  
        self.rep_started = False
        self.rep_failed = False
        self.failure_reason = None
        self.reached_up_state = False
        self.feedback = "Ready"
        self.form_status = "good"
        self.last_sound_time = {}        
        # Performance optimization
        self.frame_skip_count = 0
        self.frame_skip_interval = 2
        self.gc_counter = 0
        self.gc_interval = 100
        self.last_processed_frame = None
    def cleanup(self):
        """Cleanup resources when tracker is being destroyed"""
        try:
            if self.music_playing:
                self.stop_background_music()
            
            if self.pygame_initialized:
                pygame.mixer.quit()
                self.pygame_initialized = False
            
            if hasattr(self, 'pose'):
                self.pose.close()
        except Exception as e:
            print(f"Error during cleanup: {e}")

    
    def initialize_audio(self):
        if not self.pygame_initialized:
            try:
                pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
                pygame.mixer.set_num_channels(8)
                self.pygame_initialized = True
            except:
                print("Warning: Could not initialize pygame mixer")
                return False
        
        if not self.sounds_loaded and self.pygame_initialized:
            try:
                self.sound_success = pygame.mixer.Sound(self.success_sound_path)
                self.sound_too_high = pygame.mixer.Sound(self.too_high_sound_path)
                self.sound_bad_form = pygame.mixer.Sound(self.bad_form_sound_path)
                self.sound_try_again = pygame.mixer.Sound(self.try_again_sound_path)
                self.sounds_loaded = True
                self.sound_enabled = True
            except Exception as e:
                print(f"Could not load sound files: {e}")
                return False
        
        return True
    
    def start_background_music(self):
        if not self.initialize_audio():
            return
        
        if not self.music_loaded:
            try:
                pygame.mixer.music.load(self.background_sound_path)
                pygame.mixer.music.set_volume(0.3)
                self.music_loaded = True
            except Exception as e:
                print(f"Could not load background music: {e}")
                return
        
        if not self.music_playing:
            try:
                pygame.mixer.music.play(-1)
                self.music_playing = True
            except Exception as e:
                print(f"Could not play background music: {e}")
    
    def stop_background_music(self):
        if self.music_playing:
            pygame.mixer.music.stop()
            self.music_playing = False
    
    def play_event_sound(self, sound, sound_type):
        if not self.sound_enabled or not self.sounds_loaded:
            return
        
        now = time.time()
        if sound_type not in self.last_sound_time or \
           now - self.last_sound_time[sound_type] > self.sound_cooldown:
            try:
                sound.play()
                self.last_sound_time[sound_type] = now
            except Exception as e:
                print(f"Could not play sound: {e}")
    
    def calculate_angle(self, a, b, c):
#Angle between 3 points
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        
        if angle > 180.0:
            angle = 360 - angle
            
        return angle
    
    def check_form(self, landmarks, current_angle=None, current_state=None):
        try:
            # Get landmarks for wrists and shoulders
            left_wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST]
            right_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST]
            left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
            
            # Calculate distance between wrists and shoulders
            wrist_distance = abs(left_wrist.x - right_wrist.x)
            shoulder_distance = abs(left_shoulder.x - right_shoulder.x)
            
            # Condition: Arms shouldn't be too close (narrower than 70% of shoulder width)
            if wrist_distance < shoulder_distance * 0.7:
                return False, "Too narrow - widen grip"
            
            return True, "Good form"
        except:
            return True, ""

    def process_frame(self, frame):
        # Performance optimizations
        self.gc_counter += 1
        if self.gc_counter >= self.gc_interval:
            gc.collect()
            self.gc_counter = 0
        
        self.frame_skip_count += 1
        if self.frame_skip_count % self.frame_skip_interval != 0:
            if self.last_processed_frame:
                return self.last_processed_frame
        
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = self.pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        form_warning = ""
        angle = 0
        feedback = self.feedback
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # Get coordinates
            l_shoulder = [landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER].x, 
                         landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER].y]
            l_elbow = [landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW].x, 
                      landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW].y]
            l_wrist = [landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST].x, 
                      landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST].y]
            
            r_shoulder = [landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].x, 
                         landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].y]
            r_elbow = [landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW].x, 
                      landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW].y]
            r_wrist = [landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST].x, 
                      landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST].y]
            
            # Calculate angles
            angle_l = self.calculate_angle(l_shoulder, l_elbow, l_wrist)
            angle_r = self.calculate_angle(r_shoulder, r_elbow, r_wrist)
            angle = (angle_l + angle_r) / 2
            
            current_time = time.time()
            form_ok, form_feedback = self.check_form(landmarks)
            
            if not form_ok:
                form_warning = form_feedback
                self.form_status = "error"
                if self.rep_started and not self.rep_failed:
                    self.rep_failed = True
                    self.failure_reason = "bad_form"
                    if self.sounds_loaded:
                        self.play_event_sound(self.sound_bad_form, "bad_form")
                self.state = "down"
                self.up_time = None
                feedback = "Fix form: " + form_feedback
            else:
                self.form_status = "good"
                
                # Rep counting logic 
                if self.state == "down":
                    if angle > self.FULL_DOWN:  
                        self.state = "pressing"
                        feedback = "Pushing..."
                        self.rep_started = True
                
                elif self.state == "pressing":
                    if angle >= self.FULL_UP:
                        self.state = "up"
                        self.up_time = current_time
                        self.reached_up_state = True
                        feedback = "Top Position - Hold!"
                    elif angle < self.FULL_DOWN:
                        self.state = "down"
                        feedback = "Keep pushing up!"
                
                elif self.state == "up":
                    if self.up_time and (current_time - self.up_time) >= self.min_rep_time:
                        if angle < self.FULL_UP:  
                            self.state = "lowering"
                            feedback = "Lower Slowly"
                    else:
                        hold_time = current_time - self.up_time
                        feedback = f"Hold: {hold_time:.1f}/{self.min_rep_time}s"
                
                elif self.state == "lowering":
                    if angle <= self.FULL_DOWN:
                        if self.rep_started and not self.rep_failed and self.reached_up_state:
                            self.count += 1
                            feedback = f"Rep {self.count} Done!"
                            if self.sounds_loaded:
                                self.play_event_sound(self.sound_success, "success")
                        elif not self.reached_up_state:
                            feedback = "Push higher next time"
                        elif self.rep_failed and self.failure_reason:
                            feedback = f"Failed: {self.failure_reason.replace('_', ' ')}"
                        
                        self.state = "down"
                        self.rep_started = False
                        self.rep_failed = False
                        self.failure_reason = None
                        self.reached_up_state = False
            
            # Determine drawing color based on form status
            if self.form_status == "good":
                drawing_spec = self.drawing_spec_good
                angle_color = (0, 255, 0)
            elif self.form_status == "warning":
                drawing_spec = self.drawing_spec_warning
                angle_color = (0, 165, 255)
            else:  
                drawing_spec = self.drawing_spec_error
                angle_color = (0, 0, 255)
            
            # Draw landmarks with color
            self.mp_drawing.draw_landmarks(
                image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                drawing_spec, drawing_spec
            )
            
            # Draw angle text
            cv2.putText(image, f'Angle: {int(angle)}°', (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, angle_color, 2)
        
        else:
            feedback = "No pose detected - Stand in view"
        
        # Draw stats
        cv2.putText(image, f'Count: {self.count}', (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        cv2.putText(image, f'State: {self.state}', (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 140, 0), 2)
        
        cv2.putText(image, feedback, (10, 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        if form_warning:
            cv2.putText(image, form_warning, (10, 150),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Cache for frame skipping
        self.last_processed_frame = (image, self.count, feedback)
        return image, self.count, feedback


# For standalone use
if __name__ == "__main__":
    tracker = OverheadPressTracker()
    
    cap = cv2.VideoCapture(0)

    print("Press 'q' to quit")
    print("Press 'r' to reset counter")
    print("Press 'm' to toggle background music")
        
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: 
            break
            
        processed_frame, count, feedback = tracker.process_frame(frame)
        cv2.imshow('Overhead Press Tracker', processed_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            tracker.reset()
            print("Counter reset!")
        elif key == ord('m'):
            if tracker.music_playing:
                tracker.stop_background_music()
                print("Music stopped")
            else:
                tracker.start_background_music()
                print("Music started")
            
        if cv2.getWindowProperty('Overhead Press Tracker', cv2.WND_PROP_VISIBLE) < 1:
            break
    
    if tracker.music_playing:
        tracker.stop_background_music()
    cap.release()
    cv2.destroyAllWindows()