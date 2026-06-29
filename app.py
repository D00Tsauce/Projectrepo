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
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600&display=swap');

    /* 1. Base App Canvas & Moving Aurora Gradient */
    .stApp {
        background: linear-gradient(-45deg, #0052d4, #4364f7, #6fb1fc, #7bed9f);
        background-size: 400% 400%;
        animation: auroraAnimation 15s ease infinite;
        color: #ffffff;
        font-family: 'Outfit', sans-serif !important;
    }
    @keyframes auroraAnimation {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* 2. Windows Aero Glassmorphism Containers */
    [data-testid="stSidebar"], .aero-card {
        background: rgba(255, 255, 255, 0.12) !important;
        backdrop-filter: blur(20px) saturate(200%) !important;
        -webkit-backdrop-filter: blur(20px) saturate(200%) !important;
        border: 1px solid rgba(255, 255, 255, 0.3) !important;
        border-radius: 24px !important;
        box-shadow: 0 8px 32px 0 rgba(0, 114, 255, 0.2) !important;
    }
    
    [data-testid="stSidebar"] {
        border-radius: 0px 24px 24px 0px !important;
        padding-top: 2rem;
    }

    /* 3. Glossy 3D Windows Vista/Wii Styled Widgets */
    div.stAlert {
        background: rgba(255, 255, 255, 0.2) !important;
        border: 1px solid rgba(255, 255, 255, 0.4) !important;
        border-radius: 16px !important;
        backdrop-filter: blur(10px);
    }
    div.stAlert p {
        color: #ffffff !important;
        font-weight: 600;
    }

    /* 4. High-Gloss Interactive Buttons */
    div.stButton > button {
        background: linear-gradient(180deg, rgba(255,255,255,0.7) 0%, rgba(255,255,255,0) 50%, rgba(0,0,0,0.15) 100%), #00ca4e !important;
        color: white !important;
        border: 1px solid rgba(255, 255, 255, 0.5) !important;
        border-radius: 30px !important;
        padding: 0.6rem 2rem !important;
        font-weight: 600 !important;
        box-shadow: 0 8px 20px rgba(0,0,0,0.15), inset 0 2px 4px rgba(255,255,255,0.6) !important;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    }
    div.stButton > button:hover {
        transform: translateY(-3px) scale(1.02);
        box-shadow: 0 12px 25px rgba(0, 202, 78, 0.4), inset 0 2px 4px rgba(255,255,255,0.8) !important;
    }

    /* 5. Custom Live ECE Metric Widgets (Vista-Style Sidebar HUD) */
    .hud-metric {
        background: rgba(0, 0, 0, 0.15);
        border-radius: 14px;
        padding: 12px;
        margin-bottom: 10px;
        border-left: 4px solid #7bed9f;
    }

    /* 6. Titles & Accent Lighting */
    h1, h2, h3 {
        text-shadow: 0 4px 12px rgba(0,0,0,0.25);
        letter-spacing: -0.5px;
    }
            
    /* High-Contrast Readability Outline Overrides */
    [data-testid="stSidebar"] h4,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] small {
        font-weight: 800 !important;
        color: #ffffff !important;
        text-shadow: -1px -1px 0 #000, 1px -1px 0 #000, -1px 1px 0 #000, 1px 1px 0 #000, 0px 2px 5px rgba(0,0,0,0.9) !important;
    }
    </style>
""", unsafe_allow_html=True)

# Custom Frutiger Aero HUD Component in Sidebar
st.sidebar.markdown("""
    <div style='margin-top: 2rem; margin-bottom: 1rem;'>
        <h4 style='color: white; font-weight: 800; text-shadow: -1px -1px 0 #000, 1px -1px 0 #000, -1px 1px 0 #000, 1px 1px 0 #000;'>📡 Telemetry Matrix</h4>
        <div class='hud-metric' style='background: rgba(0, 0, 0, 0.4); border: 1px solid rgba(255,255,255,0.25);'>
            <small style='color: #ffffff; display:block; font-weight: 800; text-shadow: -1px -1px 0 #000, 1px -1px 0 #000, -1px 1px 0 #000, 1px 1px 0 #000;'>Core Engine Status</small>
            <strong style='color: #7bed9f; font-weight: 800; text-shadow: -1px -1px 0 #000, 1px -1px 0 #000, -1px 1px 0 #000, 1px 1px 0 #000;'>🟢 Nominal (Active)</strong>
        </div>
        <div class='hud-metric' style='background: rgba(0, 0, 0, 0.4); border: 1px solid rgba(255,255,255,0.25);'>
            <small style='color: #ffffff; display:block; font-weight: 800; text-shadow: -1px -1px 0 #000, 1px -1px 0 #000, -1px 1px 0 #000, 1px 1px 0 #000;'>Processing Framework</small>
            <strong style='color: #00c6ff; font-weight: 800; text-shadow: -1px -1px 0 #000, 1px -1px 0 #000, -1px 1px 0 #000, 1px 1px 0 #000;'>YOLOv8n + OpenCV Chain</strong>
        </div>
        <div class='hud-metric' style='background: rgba(0, 0, 0, 0.4); border: 1px solid rgba(255,255,255,0.25);'>
            <small style='color: #ffffff; display:block; font-weight: 800; text-shadow: -1px -1px 0 #000, 1px -1px 0 #000, -1px 1px 0 #000, 1px 1px 0 #000;'>Target Hardware Profile</small>
            <strong style='color: #ffb8b8; font-weight: 800; text-shadow: -1px -1px 0 #000, 1px -1px 0 #000, -1px 1px 0 #000, 1px 1px 0 #000;'>Edge-CPU Realtime Array</strong>
        </div>
    </div>
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