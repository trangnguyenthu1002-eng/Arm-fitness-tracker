import streamlit as st
from streamlit_webrtc import webrtc_streamer, RTCConfiguration, WebRtcMode
import av
import cv2
import time
from datetime import datetime  

# Import Class
from BicepCurl import BicepsCurlTracker
from LateralRaise import LateralRaiseTracker
from overhead_press import OverheadPressTracker
from instruction import show_instructions

from database import init_db, save_session, get_history 
init_db()

st.set_page_config(page_title="Arm Fitness Coach", layout="wide")
st.title("Arm Fitness Trainer")

exercise_option = st.sidebar.selectbox(
    "Chọn bài tập:", 
    ["Bicep Curl", "Lateral Raise", "Overhead Press"]
)

st.sidebar.markdown("---")
if st.sidebar.button("View Instructions", use_container_width=True):
    st.session_state.show_instructions = not st.session_state.get('show_instructions', False)

st.sidebar.markdown("---")
st.sidebar.subheader("Workout History")
if st.sidebar.checkbox("Show Progress"):
    history = get_history()
    if history:
        for exercise, reps, time in history:
            st.sidebar.write(f"**{exercise}**: {reps} reps ({time[:16]})")
    else:
        st.sidebar.write("No sessions recorded yet.")

# Initialize session state
if 'tracker' not in st.session_state or st.session_state.get('current_exercise') != exercise_option:
    # Clean up old tracker if exists
    if 'tracker' in st.session_state:
        old_tracker = st.session_state.tracker
        
        # Stop any playing music
        if hasattr(old_tracker, 'stop_background_music'):
            try:
                old_tracker.stop_background_music()
            except:
                pass
        
        # Close MediaPipe pose instance
        if hasattr(old_tracker, 'pose'):
            try:
                old_tracker.pose.close()
            except:
                pass
        
        # Quit pygame mixer to avoid conflicts
        if hasattr(old_tracker, 'pygame_initialized') and old_tracker.pygame_initialized:
            try:
                import pygame
                pygame.mixer.quit()
                time.sleep(0.1)  # Small delay to ensure cleanup
            except:
                pass
    
    # Create new tracker
    if exercise_option == "Bicep Curl":
        st.session_state.tracker = BicepsCurlTracker()
    elif exercise_option == "Lateral Raise":
        st.session_state.tracker = LateralRaiseTracker()
    else:
        st.session_state.tracker = OverheadPressTracker()
    
    st.session_state.current_exercise = exercise_option
    st.session_state.music_started = False
    st.session_state.stream_active = False
    st.session_state.workout_completed = False
    st.session_state.show_balloons = False
    st.session_state.restart_key = st.session_state.get('restart_key', 0) + 1

tracker = st.session_state.tracker

def stop_workout():
    if hasattr(tracker, 'stop_background_music'):
        tracker.stop_background_music()
    
    if hasattr(tracker, '_music_started_flag'):
        tracker._music_started_flag = False
    if hasattr(tracker, 'music_playing'):
        tracker.music_playing = False
    
    # Store final count before resetting
    final_count = tracker.count 
    st.session_state.final_count = final_count

    save_session(st.session_state.current_exercise, final_count) 
    
    # Update session state
    st.session_state.music_started = False
    st.session_state.stream_active = False
    st.session_state.workout_completed = True
    st.session_state.show_balloons = True

# Function to restart workout
def restart_workout():
    if exercise_option == "Bicep Curl":
        st.session_state.tracker = BicepsCurlTracker()
    elif exercise_option == "Lateral Raise":
        st.session_state.tracker = LateralRaiseTracker()
    else:
        st.session_state.tracker = OverheadPressTracker()
    
    tracker = st.session_state.tracker
    
    # Reset all state
    st.session_state.music_started = False
    st.session_state.stream_active = True
    st.session_state.workout_completed = False
    st.session_state.show_balloons = False
    st.session_state.current_count = 0
    st.session_state.current_feedback = "Ready!"
    st.session_state.restart_key += 1

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    if not hasattr(tracker, '_music_started_flag'):
        tracker._music_started_flag = False
    
    if not tracker._music_started_flag and st.session_state.get('stream_active', False):
        if hasattr(tracker, 'start_background_music'):
            tracker.start_background_music()
        tracker._music_started_flag = True
        st.session_state.music_started = True
    
    # Process frame
    result = tracker.process_frame(img)
    processed_img = result[0]
    count = result[1]
    feedback = result[2]
    
    st.session_state.current_feedback = feedback
    st.session_state.current_count = count
    
    return av.VideoFrame.from_ndarray(processed_img, format="bgr24")

# Main interface 
col_video, col_stats = st.columns([3, 1])

with col_video:
    if st.session_state.get('show_balloons', False):
        st.balloons()
        st.session_state.show_balloons = False
        st.rerun()
    
    if st.session_state.get('workout_completed', False):
        if st.session_state.get('show_balloons', False):
            st.balloons()
            st.session_state.show_balloons = False
        
        st.success(f"Workout hoàn thành! Tổng số reps: {st.session_state.get('final_count', 0)}")
        st.info("Nhấn 'Bắt đầu lại' để tập tiếp.")
        
        if st.button("Bắt đầu lại", type="primary", key="restart_btn"):
            restart_workout()
            st.rerun()
    
    elif st.session_state.get('stream_active', False):
        ctx = webrtc_streamer(
            key=f"fitness-stream-{exercise_option}-{st.session_state.restart_key}",
            video_frame_callback=video_frame_callback,
            rtc_configuration=RTCConfiguration(
                {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
            ),
            media_stream_constraints={
                "video": {
                    "width": {"ideal": 640, "max": 640},
                    "height": {"ideal": 480, "max": 480},
                    "frameRate": {"ideal": 15, "max": 20}
                },
                "audio": False
            },
            async_processing=True,
            mode=WebRtcMode.SENDRECV,
        )
        
        # Monitor stream state
        if ctx.state.playing:
            pass
        else:
            if st.session_state.get('music_started', False):
                if hasattr(tracker, 'stop_background_music'):
                    tracker.stop_background_music()
                if hasattr(tracker, '_music_started_flag'):
                    tracker._music_started_flag = False
                st.session_state.music_started = False
    
    else:
        if st.button("Start", type="primary", use_container_width=True):
            st.session_state.stream_active = True
            st.session_state.workout_completed = False
            st.session_state.music_started = False
            tracker.reset()
            st.session_state.current_count = 0
            st.session_state.current_feedback = "Ready!"
            st.rerun()

# Show instructions 
if st.session_state.get('show_instructions', False):
    with col_video:
        st.markdown("---") 
        show_instructions(exercise_option)
        
        if st.button("Close Instructions", use_container_width=True, key="close_instructions"):
            st.session_state.show_instructions = False
            st.rerun()
        st.markdown("---")

with col_stats:
    st.subheader("Assessment")
    
    st.markdown("---")
    
    if st.session_state.get('workout_completed', False):
        current_count = st.session_state.get('final_count', 0)
        st.metric(label="Tổng số Reps", value=current_count, delta="Hoàn thành!")
        st.info(" Buổi tập đã kết thúc")
    else:
        current_count = st.session_state.get('current_count', tracker.count)
        st.metric(label="Số Reps", value=current_count)
        
        # Show feedback
        f_text = st.session_state.get('current_feedback', "Ready!")
        st.info(f"Hướng dẫn: {f_text}")
    
    if st.session_state.get('stream_active', False) and not st.session_state.get('workout_completed', False):
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Reset Counter", use_container_width=True, key="reset_counter"):
                tracker.reset()
                st.session_state.current_count = 0
                st.session_state.current_feedback = "Ready!"
                st.rerun()
        
        with col2:
            if st.session_state.get('music_started', False):
                if st.button("Stop music?", use_container_width=True, key="pause_music"):
                    if hasattr(tracker, 'stop_background_music'):
                        tracker.stop_background_music()
                    if hasattr(tracker, '_music_started_flag'):
                        tracker._music_started_flag = False
                    st.session_state.music_started = False
                    st.rerun()
            else:
                if st.button("Resume music?", use_container_width=True, key="resume_music"):
                    if hasattr(tracker, 'start_background_music'):
                        tracker.start_background_music()
                    if hasattr(tracker, '_music_started_flag'):
                        tracker._music_started_flag = True
                    st.session_state.music_started = True
                    st.rerun()
        
        if st.button("End Workout", type="primary", use_container_width=True, key="end_workout"):
            stop_workout()
            st.rerun()
    
    elif not st.session_state.get('stream_active', False) and not st.session_state.get('workout_completed', False):
        st.info("Nhấn 'Bắt đầu tập' để bắt đầu buổi tập mới.")