# Arm Fitness Tracker

AI-powered fitness application that tracks and analyzes arm workouts in real-time using MediaPipe and OpenCV.

## Features

- **Real-time Pose Estimation**: Uses MediaPipe to track joints and calculate precise joint angles
- **Exercise Detection**: Optimized for:
  - **Bicep Curls**: Tracks elbow flexion and ensures full range of motion
  - **Lateral Raises**: Monitors shoulder height and lateral movement
- **Intelligent Feedback**: Visual cues showing current state and angles
- **Progress Tracking**: Real-time rep counting

## Technology Stack

- Python 3.11+
- MediaPipe 0.9.3.0 (for pose tracking)
- Streamlit (web interface)
- OpenCV (image processing)
- Streamlit WebRTC (real-time video streaming)

## Installation

### Local Setup

1. Clone the repository:
```bash
git clone https://github.com/buivan19/Arm-fitness-tracker.git
cd Arm-fitness-tracker
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the app:
```bash
streamlit run app.py
```

## Streamlit Cloud Deployment

1. Push your code to GitHub repository
2. Go to [Streamlit Cloud](https://share.streamlit.io/)
3. Click "New app"
4. Select your repository: `buivan19/Arm-fitness-tracker`
5. Set:
   - **Main file path**: `app.py`
   - **Python version**: 3.11 (recommended)
6. Click "Deploy"

## Requirements

- Python 3.11 or 3.12 (not 3.13)
- Webcam access for real-time tracking
- Modern web browser with WebRTC support

## File Structure

- `app.py` - Main Streamlit application
- `BicepCurl.py` - Bicep curl exercise tracker
- `LateralRaise.py` - Lateral raise exercise tracker
- `requirements.txt` - Python dependencies
- `packages.txt` - System dependencies for Streamlit Cloud

## Notes

- Audio files (`.mp3`, `.wav`) are optional and may not work in Streamlit Cloud
- The app uses MediaPipe 0.9.3.0 which supports the Solutions API
- For best performance, use Python 3.11 on Streamlit Cloud


