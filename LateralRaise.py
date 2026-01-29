import cv2
import numpy as np
import mediapipe as mp
import time
import pygame
import gc

class LateralRaiseTracker:
    def __init__(self):
        self.success_sound_path = "audio/perfect.wav"
        self.background_music_path = "audio/background_music.mp3"
        self.too_high_sound_path = "audio/too_high.mp3"
        self.bad_form_sound_path = "audio/bad_form.mp3"
        self.try_again_sound_path = "audio/try_again.wav"
        
        # Pygame initialization
        self.pygame_initialized = False
        self.sounds_loaded = False
        self.sound_enabled = False
        self.music_loaded = False
        self.music_playing = False
        
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
        
        self.FULL_DOWN = 20
        self.MID_POINT = 45
        self.FULL_UP = 90     
        self.MAX_ANGLE = 120
        
        self.ELBOW_MAX_HEIGHT = 0.05

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

    def reset(self):
        self.count = 0
        self.state = "down"
        self.up_time = None
        self.min_rep_time = 0.8
        self.rep_started = False
        self.rep_failed = False
        self.failure_reason = None
        self.reached_up_state = False
        self.last_feedback = "Ready"
        self.form_status = "good"
        # Reset audio tracking
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


    def start_background_music(self):
        if not self.initialize_audio():
            return
        
        if not self.music_loaded:
            try:
                pygame.mixer.music.load(self.background_music_path)
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
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        
        if angle > 180.0:
            angle = 360 - angle
            
        return angle
    
    def check_form(self, landmarks):
        try:
            left_shoulder_y = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER].y
            left_elbow_y = landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW].y
            right_shoulder_y = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].y
            right_elbow_y = landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW].y
            
            left_bad_form = left_elbow_y < (left_shoulder_y - self.ELBOW_MAX_HEIGHT)
            right_bad_form = right_elbow_y < (right_shoulder_y - self.ELBOW_MAX_HEIGHT)
            
            if left_bad_form or right_bad_form:
                return False, "Elbows above shoulders - lower arms"
            
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
        
        feedback = "No pose detected"
        form_warning = ""
        self.form_status = "good"
        
        try:
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                
                # Get left arm
                left_shoulder = [landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                                landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER].y]
                left_elbow = [landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW].x,
                             landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW].y]
                left_hip = [landmarks[self.mp_pose.PoseLandmark.LEFT_HIP].x,
                           landmarks[self.mp_pose.PoseLandmark.LEFT_HIP].y]
                
                # Get right arm
                right_shoulder = [landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
                                 landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].y]
                right_elbow = [landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW].x,
                              landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW].y]
                right_hip = [landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP].x,
                            landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP].y]
                
                # Calculate angles
                left_angle = self.calculate_angle(left_hip, left_shoulder, left_elbow)
                right_angle = self.calculate_angle(right_hip, right_shoulder, right_elbow)
                
                angle = (left_angle + right_angle) / 2
                
                form_ok, form_feedback = self.check_form(landmarks)
                
                current_time = time.time()
                
                # Check for form violations and mark as failed
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
                    self.reached_up_state = False
                    
                elif angle > self.MAX_ANGLE:
                    form_warning = "Too high! Lower arms"
                    self.form_status = "error"
                    if self.rep_started and not self.rep_failed:
                        self.rep_failed = True
                        self.failure_reason = "too_high"
                        if self.sounds_loaded:
                            self.play_event_sound(self.sound_too_high, "too_high")
                    self.state = "down"
                    self.up_time = None
                    self.reached_up_state = False
                    
                else:
                    if self.state == "down":
                        if angle < self.FULL_DOWN:
                            if self.rep_failed and self.failure_reason:
                                feedback = f"Failed: {self.failure_reason.replace('_', ' ')}"
                                self.form_status = "warning"
                                self.rep_failed = False
                                self.failure_reason = None
                            
                            self.rep_started = True
                            self.reached_up_state = False  
                            if not feedback or feedback.startswith("Failed"):
                                feedback = "Ready to start"
                                self.form_status = "good"
                                
                        elif angle > self.MID_POINT and self.rep_started:
                            self.state = "raising"
                            feedback = "Raising arms..."
                            self.form_status = "good"
                            
                        elif angle > self.MID_POINT and not self.rep_started:
                            feedback = "Lower arms to start position first"
                            self.form_status = "warning"
                            if not self.rep_failed:
                                self.rep_failed = True
                                self.failure_reason = "try_again"
                                if self.sounds_loaded:
                                    self.play_event_sound(self.sound_try_again, "try_again")
                            
                    elif self.state == "raising":
                        if angle >= self.FULL_UP:
                            self.reached_up_state = True  
                            self.state = "up"
                            self.up_time = current_time
                            feedback = "Good! Hold at shoulder height"
                            self.form_status = "good"
                        elif angle < self.MID_POINT:
                            if not self.rep_failed:
                                self.rep_failed = True
                                self.failure_reason = "try_again"
                                feedback = "Raise higher!"
                                self.form_status = "warning"
                                if self.sounds_loaded:
                                    self.play_event_sound(self.sound_try_again, "try_again")
                            self.state = "down"
                            
                    elif self.state == "up":
                        if self.up_time and (current_time - self.up_time) >= self.min_rep_time:
                            if angle < self.MID_POINT:
                                self.state = "lowering"
                                feedback = "Lowering slowly..."
                                self.form_status = "good"
                        else:
                            hold_time = current_time - self.up_time
                            feedback = f"Hold: {hold_time:.1f}/{self.min_rep_time}s"
                            self.form_status = "good"
                            
                    elif self.state == "lowering":
                        if angle < self.FULL_DOWN:
                            if self.rep_started and not self.rep_failed and self.reached_up_state:
                                self.count += 1
                                feedback = f"Perfect! Rep {self.count}"
                                self.form_status = "good"
                                if self.sounds_loaded:
                                    self.play_event_sound(self.sound_success, "success")
                            elif not self.reached_up_state:
                                feedback = "Raise arms higher next time"
                                self.form_status = "warning"
                            elif self.rep_failed and self.failure_reason:
                                feedback = f"Failed: {self.failure_reason.replace('_', ' ')}"
                                self.form_status = "warning"
                            
                            self.state = "down"
                            self.rep_started = False
                            self.rep_failed = False
                            self.failure_reason = None
                            self.reached_up_state = False
                        elif angle > self.MID_POINT:
                            self.state = "raising"
                            feedback = "Complete the lowering!"
                            self.form_status = "warning"
                
                # Determine color based on form status
                if self.form_status == "good":
                    drawing_spec = self.drawing_spec_good
                    angle_color = (0, 255, 0)
                elif self.form_status == "warning":
                    drawing_spec = self.drawing_spec_warning
                    angle_color = (0, 165, 255)
                else:  # error
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
                
        except Exception as e:
            feedback = "Processing..."
        
        # Draw statistics
        cv2.putText(image, f'Count: {self.count}', (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.putText(image, f'State: {self.state}', (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 140, 0), 2)
        
        cv2.putText(image, feedback, (10, 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        if form_warning:
            cv2.putText(image, form_warning, (10, 150),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        self.last_feedback = feedback
        # Cache for frame skipping
        self.last_processed_frame = (image, self.count, feedback)
        
        return image, self.count, feedback, self.state


# For standalone use
if __name__ == "__main__":
    tracker = LateralRaiseTracker()
    
    cap = cv2.VideoCapture(0)

    print("Press 'q' to quit")
    print("Press 'r' to reset counter")
    print("Press 'm' to toggle background music")
        
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: 
            break
            
        processed_frame, count, feedback, state = tracker.process_frame(frame)
        cv2.imshow('Lateral Raise Tracker', processed_frame)
        
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
            
        if cv2.getWindowProperty('Lateral Raise Tracker', cv2.WND_PROP_VISIBLE) < 1:
            break
    
    if tracker.music_playing:
        tracker.stop_background_music()
    cap.release()
    cv2.destroyAllWindows()