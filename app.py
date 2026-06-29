import streamlit as st
import cv2
import numpy as np
import tempfile
from src.object_detector import ADASObjectDetector
from src.lane_detector import ADASLaneDetector
# Page configuration
st.set_page_config(page_title="ADAS ML Capstone Dashboard", layout="wide")
# --- FRUTIGER AERO AESTHETIC SKIN INJECTION ---
st.markdown("""
    <style>
    /* 1. Global Background: Classic Glossy Blue-to-Green Aurora Gradient */
    .stApp {
        background: linear-gradient(135deg, #0072ff 0%, #00c6ff 50%, #7bed9f 100%);
        background-attachment: fixed;
        color: #ffffff;
    }

    /* 2. Windows Aero Glassmorphism Sidebar */
    [data-testid="stSidebar"] {
        background: rgba(255, 255, 255, 0.15) !important;
        backdrop-filter: blur(12px) saturate(180%);
        -webkit-backdrop-filter: blur(12px) saturate(180%);
        border-right: 1px solid rgba(255, 255, 255, 0.25);
    }

    /* 3. Sleek Glass Cards for Main Panels and Boxes */
    div.stAlert, div[data-testid="stMarkdownContainer"] {
        color: #1e272e;
    }
    
    /* Target the main video processing bounding block */
    .element-container {
        border-radius: 16px;
    }

    /* 4. Glossy UI Buttons (Vista/Wii Style Hover Effects) */
    div.stButton > button {
        background: linear-gradient(180deg, rgba(255,255,255,0.6) 0%, rgba(255,255,255,0) 50%, rgba(0,0,0,0.1) 100%), #2ed573 !important;
        color: white !important;
        border: 1px solid rgba(255,255,255,0.4) !important;
        border-radius: 20px !important;
        box-shadow: 0 4px 15px rgba(0,0,0,0.15), inset 0 2px 2px rgba(255,255,255,0.5) !important;
        text-shadow: 0px 1px 2px rgba(0,0,0,0.3);
        transition: all 0.3s ease;
    }
    
    div.stButton > button:hover {
        background: linear-gradient(180deg, rgba(255,255,255,0.8) 0%, rgba(255,255,255,0.1) 50%, rgba(0,0,0,0.15) 100%), #26af5f !important;
        transform: scale(1.03);
        box-shadow: 0 6px 20px rgba(46, 213, 115, 0.4) !important;
    }

    /* 5. Typography Customizations */
    h1, h2, h3, h4, label {
        color: #ffffff !important;
        font-family: 'Segoe UI', -apple-system, sans-serif !important;
        text-shadow: 0 2px 4px rgba(0,0,0,0.2);
        font-weight: 600 !important;
    }
    
    /* Make subheaders stand out with bright energy */
    .stMarkdown p {
        color: #f1f2f6;
    }
    </style>
""", unsafe_allow_html=True)
# --- END OF SKIN INJECTION ---

st.title("🚗 Advanced Driver Assistance System (ADAS) Portal")
st.subheader("Machine Learning & CV Integration for ECE Capstone Project")
st.markdown("Developed by: **N G VINAYAK, Prabhav P, Pranav PK**")

# Initialize backends
@st.cache_resource
def load_models():
    return ADASObjectDetector(), ADASLaneDetector()

obj_detector, lane_detector = load_models()

# Sidebar configuration
st.sidebar.header("System Settings")
enable_object_detection = st.sidebar.checkbox("Enable YOLOv8 Obstacle Tracking", value=True)
enable_lane_detection = st.sidebar.checkbox("Enable DSP Lane Detection", value=True)

uploaded_file = st.sidebar.file_uploader("Upload Dashcam Footage (.mp4)", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    # Save uploaded file to a temporary location to stream it via OpenCV
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    
    video_cap = cv2.VideoCapture(tfile.name)
    st_frame = st.empty()
    
    st.sidebar.success("Processing Video...")
    
    while video_cap.isOpened():
        ret, frame = video_cap.read()
        if not ret:
            break
            
        # Resize for smooth processing speed
        frame = cv2.resize(frame, (854, 480))
        
        # Apply DSP Lane Tracking
        if enable_lane_detection:
            frame = lane_detector.detect_lanes(frame)
            
        # Apply AI Object Tracking
        if enable_object_detection:
            frame = obj_detector.detect_obstacles(frame)
            
        # Streamlit web displays frame in RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        st_frame.image(frame_rgb, channels="RGB", use_container_width=True)
        
    video_cap.release()
    st.sidebar.info("Video Processing Finished!")
else:
    st.info("👈 Please upload a vehicle dashcam video file in the sidebar panel to test the system live.")