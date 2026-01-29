import cv2
import numpy as np
import mediapipe as mp
import time
import pygame
import gc

class BicepsCurlTracker:
    def __init__(self):
        self.success_sound_path = "audio/perfect.wav"
        self.retry_sound_path = "audio/try_again.wav"
        self.too_high_sound_path = "audio/curl_more.wav"
        self.too_low_sound_path = "audio/extend.wav"
        self.background_sound_path = "audio/background_music.mp3"

        self.pygame_initialized = False
        self.sounds_loaded = False
        self.music_loaded = False
        self.music_playing = False

        self.last_sound_time = {}
        self.sound_cooldown = {
            'retry': 0.8,
            'form': 1.0,
            'success': 0.5,
            'too_high': 0.7,
            'too_low': 0.7
        }
        
        self.last_form_state = "good"
        self.form_failed_time = 0
        self.form_feedback_cooldown = 1.0
    
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils

        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1, # Balanced mode
            smooth_landmarks=True, # Reduce jitter
            enable_segmentation=False, # Disable unused feature
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )

        self.count = 0
        self.state = "down"
        self.up_time = None
        self.min_rep_time = 0.5

        self.FULL_DOWN = 160
        self.MID_POINT = 100
        self.FULL_UP = 60
        self.MIN_ANGLE = 30

        self.ELBOW_DRIFT = 0.08

        # Performance optimization
        self.frame_skip_count = 0
        self.frame_skip_interval = 2  # Process every 2nd frame
        self.gc_counter = 0
        self.gc_interval = 100
        self.last_processed_frame = None

        
        self.was_attempting_rep = False
        self.last_retry_state = None

    def reset(self):
        self.count = 0
        self.state = "down"
        self.up_time = None
        self.last_form_state = "good"
        self.form_failed_time = 0
        self.was_attempting_rep = False
        self.last_retry_state = None
        self.last_sound_time = {}
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
                self.sound_retry = pygame.mixer.Sound(self.retry_sound_path)
                self.sound_too_high = pygame.mixer.Sound(self.too_high_sound_path)
                self.sound_too_low = pygame.mixer.Sound(self.too_low_sound_path)
                self.sounds_loaded = True
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
        if not self.sounds_loaded:
            return
        
        # Prevent channel buildup
        if pygame.mixer.get_busy():
            busy_channels = sum(1 for i in range(pygame.mixer.get_num_channels()) 
                              if pygame.mixer.Channel(i).get_busy())
            if busy_channels > 6:
                pygame.mixer.Channel(0).stop()
        
        now = time.time()
        
        if sound_type in self.sound_cooldown:
            cooldown = self.sound_cooldown[sound_type]
        else:
            cooldown = 0.5
        
        if sound_type not in self.last_sound_time or \
           now - self.last_sound_time[sound_type] > cooldown:
            try:
                channel = pygame.mixer.find_channel()
                if channel:
                    channel.play(sound)
                else:
                    sound.play()
                self.last_sound_time[sound_type] = now
                return True
            except Exception as e:
                print(f"Could not play sound: {e}")
        return False

    def calculate_angle(self, a, b, c):
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)

        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - \
                  np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        if angle > 180:
            angle = 360 - angle
        return angle

    def check_form(self, landmarks):
        try:
            ls = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER].x
            le = landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW].x
            rs = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].x
            re = landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW].x

            if abs(le - ls) > self.ELBOW_DRIFT or abs(re - rs) > self.ELBOW_DRIFT:
                return False, "Keep elbows close"

            return True, ""
        except:
            return True, ""

    def process_frame(self, frame):
        # Garbage collection for performance
        self.gc_counter += 1
        if self.gc_counter >= self.gc_interval:
            gc.collect()
            self.gc_counter = 0
        
        # Frame skipping for better performance
        self.frame_skip_count += 1
        if self.frame_skip_count % self.frame_skip_interval != 0:
            if self.last_processed_frame:
                return self.last_processed_frame
        
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = self.pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        feedback = ""
        current_time = time.time()

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark

            # Left arm
            l_shoulder = [lm[self.mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                          lm[self.mp_pose.PoseLandmark.LEFT_SHOULDER].y]
            l_elbow = [lm[self.mp_pose.PoseLandmark.LEFT_ELBOW].x,
                       lm[self.mp_pose.PoseLandmark.LEFT_ELBOW].y]
            l_wrist = [lm[self.mp_pose.PoseLandmark.LEFT_WRIST].x,
                       lm[self.mp_pose.PoseLandmark.LEFT_WRIST].y]

            # Right arm
            r_shoulder = [lm[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
                          lm[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].y]
            r_elbow = [lm[self.mp_pose.PoseLandmark.RIGHT_ELBOW].x,
                       lm[self.mp_pose.PoseLandmark.RIGHT_ELBOW].y]
            r_wrist = [lm[self.mp_pose.PoseLandmark.RIGHT_WRIST].x,
                       lm[self.mp_pose.PoseLandmark.RIGHT_WRIST].y]

            left_angle = self.calculate_angle(l_shoulder, l_elbow, l_wrist)
            right_angle = self.calculate_angle(r_shoulder, r_elbow, r_wrist)
            angle = (left_angle + right_angle) / 2

            form_ok, form_msg = self.check_form(lm)
            
            # Track if user is actively attempting a rep
            is_in_motion = self.state in ["curling", "up", "lowering"]
            just_started_rep = (self.state == "curling" and 
                               (self.last_retry_state != "curling" or 
                                current_time - self.form_failed_time > 1.0))
            
            if not form_ok:
                feedback = form_msg
                
                if is_in_motion:
                    if self.last_form_state == "good" or just_started_rep:
                        if self.sounds_loaded:
                            sound_played = self.play_event_sound(self.sound_retry, 'retry')
                            if sound_played:
                                self.last_retry_state = self.state
                                self.form_failed_time = current_time
                    elif current_time - self.form_failed_time > self.form_feedback_cooldown:
                        if self.sounds_loaded:
                            sound_played = self.play_event_sound(self.sound_retry, 'retry')
                            if sound_played:
                                self.form_failed_time = current_time
                
                self.last_form_state = "bad"
                self.state = "down"
                self.up_time = None

            else:
                if self.last_form_state == "bad":
                    self.last_retry_state = None
                
                self.last_form_state = "good"

                if self.state == "down":
                    if angle < self.MID_POINT:
                        self.state = "curling"
                        feedback = "Curling..."

                elif self.state == "curling":
                    if angle > self.FULL_UP and angle > self.MID_POINT:
                        feedback = "Curl higher"
                        
                        if self.sounds_loaded and self.last_retry_state != "curling_too_high":
                            sound_played = self.play_event_sound(self.sound_too_high, 'too_high')
                            if sound_played:
                                self.last_retry_state = "curling_too_high"
                    
                    elif angle <= self.FULL_UP:
                        self.state = "up"
                        self.up_time = current_time
                        feedback = "Squeeze"
                        self.last_retry_state = None

                elif self.state == "up":
                    if (current_time - self.up_time) >= self.min_rep_time:
                        if angle > self.MID_POINT:
                            self.state = "lowering"
                            feedback = "Lowering"
                            self.last_retry_state = None

                elif self.state == "lowering":
                    if angle < self.FULL_DOWN and angle < self.MID_POINT:
                        feedback = "Extend arm fully"
                        
                        if self.sounds_loaded and self.last_retry_state != "lowering_too_low":
                            sound_played = self.play_event_sound(self.sound_too_low, 'too_low')
                            if sound_played:
                                self.last_retry_state = "lowering_too_low"
                    
                    elif angle >= self.FULL_DOWN:
                        self.count += 1
                        self.state = "down"
                        feedback = f"Rep {self.count}"
                        
                        if self.sounds_loaded:
                            self.play_event_sound(self.sound_success, 'success')
                        self.last_retry_state = None

            cv2.putText(image, f'Angle: {int(angle)} deg', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

            self.mp_drawing.draw_landmarks(
                image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS
            )

        else:
            feedback = "No pose detected - Stand in view"
            self.last_form_state = "good"
            self.last_retry_state = None

        cv2.putText(image, f'Count: {self.count}', (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 120, 0), 2)
        cv2.putText(image, f'State: {self.state}', (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 140, 0), 2)
        cv2.putText(image, feedback, (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 150), 2)

        return image, self.count, feedback


if __name__ == "__main__":
    tracker = BicepsCurlTracker()
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image, count, feedback = tracker.process_frame(frame)
        cv2.imshow("Biceps Curl Tracker", image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if cv2.getWindowProperty("Biceps Curl Tracker", cv2.WND_PROP_VISIBLE) < 1:
            break
        
    if tracker.music_playing:
        tracker.stop_background_music()
    if tracker.pygame_initialized:
        pygame.mixer.quit()
    cap.release()
    cv2.destroyAllWindows()