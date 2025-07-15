# app.py - Enhanced version with modern UI and animations
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
import sys
import traceback
import time

# Add current directory to path
sys.path.append('.')

# Page configuration
st.set_page_config(
    page_title="AI Object Detection Studio",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS with modern design and animations
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        min-height: 100vh;
    }
    
    .stApp {
        background: transparent;
    }
    
    .main-header {
        font-size: 3.5rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(45deg, #667eea, #764ba2, #f093fb);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        animation: gradient-shift 3s ease-in-out infinite;
        text-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }
    
    @keyframes gradient-shift {
        0%, 100% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
    }
    
    .subtitle {
        text-align: center;
        font-size: 1.3rem;
        color: #ffffff;
        margin-bottom: 3rem;
        opacity: 0.9;
        font-weight: 300;
    }
    
    .glass-container {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        margin: 1rem 0;
        transition: all 0.3s ease;
    }
    
    .glass-container:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 40px rgba(0, 0, 0, 0.4);
    }
    
    .stButton > button {
        width: 100%;
        margin-top: 1rem;
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 15px;
        padding: 0.8rem 2rem;
        font-weight: 600;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6);
        background: linear-gradient(45deg, #764ba2, #667eea);
    }
    
    .stButton > button:active {
        transform: translateY(0px);
    }
    
    .detection-info {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        padding: 2rem;
        border-radius: 20px;
        margin: 2rem 0;
        border: 1px solid rgba(255, 255, 255, 0.3);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        animation: slideInUp 0.5s ease-out;
    }
    
    @keyframes slideInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .error-box {
        background: rgba(239, 68, 68, 0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(239, 68, 68, 0.3);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        animation: shake 0.5s ease-in-out;
    }
    
    @keyframes shake {
        0%, 100% { transform: translateX(0); }
        25% { transform: translateX(-5px); }
        75% { transform: translateX(5px); }
    }
    
    .success-box {
        background: rgba(34, 197, 94, 0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(34, 197, 94, 0.3);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        animation: pulse 1s ease-in-out;
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.02); }
    }
    
    .sidebar .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 10px;
    }
    
    .sidebar .stSlider > div > div {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 10px;
    }
    
    .stat-card {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        margin: 0.5rem;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        transition: all 0.3s ease;
    }
    
    .stat-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.5);
    }
    
    .stat-number {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .stat-label {
        font-size: 1rem;
        opacity: 0.9;
        font-weight: 500;
    }
    
    .feature-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        transition: all 0.3s ease;
        text-align: center;
    }
    
    .feature-card:hover {
        transform: translateY(-10px);
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
    }
    
    .feature-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
    }
    
    .developer-card {
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white;
    padding: 2rem;
    border-radius: 20px;
    text-align: center;
    margin: 2rem 0;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    animation: fadeInUp 0.8s ease-out;
}

@keyframes fadeInUp {
    from {
        opacity: 0;
        transform: translateY(50px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

.social-links {
    display: flex;
    justify-content: center;
    gap: 1rem;
    margin-top: 1.5rem;
}

.social-link {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    background: rgba(255, 255, 255, 0.2);
    backdrop-filter: blur(10px);
    padding: 0.8rem 1.5rem;
    border-radius: 25px;
    color: white;
    text-decoration: none;
    transition: all 0.3s ease;
    border: 1px solid rgba(255, 255, 255, 0.3);
}

.social-link:hover {
    transform: translateY(-3px);
    background: rgba(255, 255, 255, 0.3);
    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.2);
}

.loading-spinner {
    display: flex;
    justify-content: center;
    align-items: center;
    padding: 2rem;
}
    .spinner {
        width: 40px;
        height: 40px;
        border: 4px solid rgba(255, 255, 255, 0.3);
        border-top: 4px solid #667eea;
        border-radius: 50%;
        animation: spin 1s linear infinite;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    .tabs-container {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 0.5rem;
        margin: 2rem 0;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 1rem 2rem;
        color: white;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(255, 255, 255, 0.2);
        transform: translateY(-2px);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(45deg, #667eea, #764ba2) !important;
        color: white !important;
    }
    
    .image-container {
        border-radius: 20px;
        overflow: hidden;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease;
    }
    
    .image-container:hover {
        transform: scale(1.02);
        box-shadow: 0 15px 40px rgba(0, 0, 0, 0.4);
    }
    
    .progress-bar {
        background: linear-gradient(45deg, #667eea, #764ba2);
        height: 4px;
        border-radius: 2px;
        animation: progress 2s ease-in-out infinite;
    }
    
    @keyframes progress {
        0% { width: 0%; }
        50% { width: 100%; }
        100% { width: 0%; }
    }
    
    .floating-action {
        position: fixed;
        bottom: 2rem;
        right: 2rem;
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 50%;
        width: 60px;
        height: 60px;
        font-size: 1.5rem;
        box-shadow: 0 4px 20px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
        z-index: 1000;
    }
    
    .floating-action:hover {
        transform: scale(1.1);
        box-shadow: 0 8px 30px rgba(102, 126, 234, 0.6);
    }
    
    .stFileUploader > div {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border: 2px dashed rgba(255, 255, 255, 0.3);
        border-radius: 20px;
        padding: 2rem;
        transition: all 0.3s ease;
    }
    
    .stFileUploader > div:hover {
        border-color: #667eea;
        background: rgba(255, 255, 255, 0.15);
    }
    
    .metric-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 2rem 0;
    }
    
    .stCamera > div {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 1rem;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .sidebar .stMarkdown {
        color: white;
    }
    
    .sidebar {
        background: rgba(0, 0, 0, 0.2);
        backdrop-filter: blur(20px);
    }
    
    .sidebar .stSelectbox label,
    .sidebar .stSlider label {
        color: white !important;
        font-weight: 500;
    }
    
    .version-badge {
        position: fixed;
        top: 1rem;
        right: 1rem;
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        padding: 0.5rem 1rem;
        border-radius: 20px;
        color: white;
        font-size: 0.8rem;
        font-weight: 500;
        z-index: 1000;
    }
    
    .stExpander > div {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .stAlert {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .stWarning {
        background: rgba(245, 158, 11, 0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(245, 158, 11, 0.3);
    }
    
    .stError {
        background: rgba(239, 68, 68, 0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(239, 68, 68, 0.3);
    }
    
    .stSuccess {
        background: rgba(34, 197, 94, 0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(34, 197, 94, 0.3);
    }
    
    .stInfo {
        background: rgba(59, 130, 246, 0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(59, 130, 246, 0.3);
    }
    
    .header-container {
        text-align: center;
        padding: 2rem 0;
        margin-bottom: 3rem;
    }
    
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 2rem;
        margin: 2rem 0;
    }
    
    @media (max-width: 768px) {
        .main-header {
            font-size: 2.5rem;
        }
        
        .subtitle {
            font-size: 1.1rem;
        }
        
        .glass-container {
            padding: 1rem;
        }
        
        .feature-grid {
            grid-template-columns: 1fr;
        }
    }
</style>
""", unsafe_allow_html=True)

def check_setup():
    """Check if the app setup is correct"""
    issues = []
    
    # Check if utils directory exists
    if not os.path.exists('utils'):
        issues.append("❌ 'utils' directory not found")
    
    # Check if __init__.py exists
    if not os.path.exists('utils/__init__.py'):
        issues.append("❌ 'utils/__init__.py' file not found")
    
    # Check if detection.py exists
    if not os.path.exists('utils/detection.py'):
        issues.append("❌ 'utils/detection.py' file not found")
    
    # Try importing required modules
    try:
        from ultralytics import YOLO
    except ImportError:
        issues.append("❌ ultralytics package not installed")
    
    try:
        import cv2
    except ImportError:
        issues.append("❌ opencv-python package not installed")
    
    return issues

def setup_directories():
    """Create required directories"""
    os.makedirs('utils', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('uploads', exist_ok=True)
    
    # Create __init__.py if it doesn't exist
    init_file = 'utils/__init__.py'
    if not os.path.exists(init_file):
        with open(init_file, 'w') as f:
            f.write("# Empty file to make utils a package\n")

@st.cache_resource
def load_detector():
    """Load the object detector with error handling"""
    try:
        from utils.detection import ObjectDetector
        detector = ObjectDetector()
        
        if not detector.is_model_loaded():
            st.error("❌ Model failed to load. Please check your internet connection.")
            return None
        
        return detector
    except Exception as e:
        st.error(f"❌ Error loading detector: {str(e)}")
        st.error("Please ensure all files are in place and dependencies are installed.")
        return None

def show_header():
    """Display the enhanced header"""
    st.markdown("""
        <div class="version-badge">
            🤖 AI Studio v2.0
        </div>
        <div class="header-container">
            <h1 class="main-header">🤖 AI Object Detection Studio</h1>
            <p class="subtitle">Advanced Computer Vision with YOLOv8 • Real-time Detection • Modern Interface</p>
        </div>
    """, unsafe_allow_html=True)

def main():
    show_header()
    
    # Check setup
    setup_issues = check_setup()
    if setup_issues:
        st.markdown('<div class="glass-container">', unsafe_allow_html=True)
        st.error("⚠️ **Setup Issues Found:**")
        for issue in setup_issues:
            st.write(issue)
        
        st.info("🔧 **Quick Setup Guide:**")
        st.code("""
# 1. Create required directories and files
mkdir -p utils models uploads
touch utils/__init__.py

# 2. Install required packages
pip install streamlit ultralytics opencv-python pillow

# 3. Copy the detection.py file to utils/ directory

# 4. Run the application
streamlit run app.py
        """, language="bash")
        st.markdown('</div>', unsafe_allow_html=True)
        return
    
    # Setup directories
    setup_directories()
    
    # Load detector with enhanced loading animation
    with st.spinner("🔄 Initializing AI Models..."):
        detector = load_detector()
    
    if detector is None:
        st.markdown('<div class="error-box">', unsafe_allow_html=True)
        st.error("❌ Failed to initialize object detector. Please check the setup.")
        st.markdown('</div>', unsafe_allow_html=True)
        return
    
    # Enhanced sidebar with modern styling
    st.sidebar.markdown("## ⚙️ AI Configuration")
    
    # Model selection with descriptions
    model_options = {
        "yolov8n.pt": "⚡ Nano - Ultra Fast",
        "yolov8s.pt": "🚀 Small - Balanced",
        "yolov8m.pt": "🎯 Medium - High Accuracy",
        "yolov8l.pt": "🔥 Large - Maximum Precision"
    }
    
    model_choice = st.sidebar.selectbox(
        "🧠 Select AI Model",
        list(model_options.keys()),
        format_func=lambda x: model_options[x],
        help="Choose the model that best fits your needs"
    )
    
    # Enhanced confidence slider
    conf_threshold = st.sidebar.slider(
        "🎚️ Detection Confidence",
        min_value=0.1,
        max_value=1.0,
        value=0.3,
        step=0.05,
        help="Higher values = more confident detections"
    )
    
    # Performance metrics
    st.sidebar.markdown("### 📊 Performance Metrics")
    perf_col1, perf_col2 = st.sidebar.columns(2)
    with perf_col1:
        st.metric("Model", model_choice.replace('.pt', '').upper())
    with perf_col2:
        st.metric("Confidence", f"{conf_threshold:.1%}")
    
    # Update detector settings
    try:
        detector.update_model(model_choice)
        detector.set_confidence(conf_threshold)
    except Exception as e:
        st.sidebar.error(f"❌ Error updating settings: {str(e)}")
        return
    
    # Enhanced tabs with modern styling
    st.markdown('<div class="tabs-container">', unsafe_allow_html=True)
    tab1, tab2, tab3, tab4 = st.tabs([
        "📁 Upload & Detect", 
        "📷 Live Camera", 
        "🎯 Batch Analysis", 
        "ℹ️ About & Developer"
    ])
    st.markdown('</div>', unsafe_allow_html=True)
    
    with tab1:
        handle_image_upload(detector)
    
    with tab2:
        handle_camera_input(detector)
    
    with tab3:
        handle_batch_analysis(detector)
    
    with tab4:
    
       show_about_and_developer()

def handle_image_upload(detector):
    """Enhanced image upload with modern UI"""
    st.markdown('<div class="glass-container">', unsafe_allow_html=True)
    st.markdown("## 📁 Upload Image for AI Analysis")
    
    uploaded_file = st.file_uploader(
        "Drop your image here or click to browse", 
        type=['jpg', 'jpeg', 'png', 'bmp', 'webp'],
        help="Supported formats: JPG, PNG, BMP, WebP"
    )
    
    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🖼️ Original Image")
            st.markdown('<div class="image-container">', unsafe_allow_html=True)
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Image info
            st.markdown("**Image Details:**")
            st.write(f"📐 Size: {image.size[0]} x {image.size[1]} pixels")
            st.write(f"💾 Format: {image.format}")
            st.write(f"📊 Mode: {image.mode}")
        
        if st.button("🚀 Start AI Detection", key="detect_upload"):
            with st.spinner("🔄 AI is analyzing your image..."):
                # Progress bar simulation
                progress_bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.01)
                    progress_bar.progress(i + 1)
                
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name
                
                try:
                    # Run detection
                    result_image, detections = detector.detect_objects(tmp_file_path)
                    
                    with col2:
                        st.markdown("### 🎯 Detection Results")
                        st.markdown('<div class="image-container">', unsafe_allow_html=True)
                        st.image(result_image, caption="AI Detection Results", use_column_width=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Enhanced detection summary
                    show_enhanced_detection_summary(detections)
                    
                    # Success message
                    st.markdown('<div class="success-box">', unsafe_allow_html=True)
                    st.success(f"✅ Detection completed! Found {len(detections)} objects.")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                except Exception as e:
                    st.markdown('<div class="error-box">', unsafe_allow_html=True)
                    st.error(f"❌ Detection failed: {str(e)}")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Show detailed error in expander
                    with st.expander("🔍 Technical Details"):
                        st.code(traceback.format_exc())
                
                finally:
                    # Clean up temporary file
                    try:
                        os.unlink(tmp_file_path)
                    except:
                        pass
    
    st.markdown('</div>', unsafe_allow_html=True)

def handle_camera_input(detector):
    """Enhanced camera input with modern UI"""
    st.markdown('<div class="glass-container">', unsafe_allow_html=True)
    st.markdown("## 📷 Live Camera Detection")
    
    st.markdown("### 📸 Capture from Camera")
    camera_input = st.camera_input("Say cheese! 📸")
    
    if camera_input is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🎥 Captured Image")
            st.markdown('<div class="image-container">', unsafe_allow_html=True)
            st.image(camera_input, caption="Live Capture", use_column_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        if st.button("🎯 Analyze Live Image", key="detect_camera"):
            with st.spinner("🔄 Processing live image..."):
                # Progress animation
                progress_bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.01)
                    progress_bar.progress(i + 1)
                
                # Save camera input temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                    tmp_file.write(camera_input.getvalue())
                    tmp_file_path = tmp_file.name
                
                try:
                    # Run detection
                    result_image, detections = detector.detect_objects(tmp_file_path)
                    
                    with col2:
                        st.markdown("### 🎯 Live Detection Results")
                        st.markdown('<div class="image-container">', unsafe_allow_html=True)
                        st.image(result_image, caption="Real-time Detection", use_column_width=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Enhanced detection summary
                    show_enhanced_detection_summary(detections)
                    
                    # Success message
                    st.markdown('<div class="success-box">', unsafe_allow_html=True)
                    st.success(f"✅ Live detection completed! Found {len(detections)} objects.")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                except Exception as e:
                    st.markdown('<div class="error-box">', unsafe_allow_html=True)
                    st.error(f"❌ Live detection failed: {str(e)}")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    with st.expander("🔍 Technical Details"):
                        st.code(traceback.format_exc())
                
                finally:
                    try:
                        os.unlink(tmp_file_path)
                    except:
                        pass
    
    st.markdown('</div>', unsafe_allow_html=True)

def handle_batch_analysis(detector):
    """New batch analysis feature"""
    st.markdown('<div class="glass-container">', unsafe_allow_html=True)
    st.markdown("## 🎯 Batch Image Analysis")
    
    st.info("📊 Upload multiple images for batch processing")
    
    uploaded_files = st.file_uploader(
        "Select multiple images for batch analysis",
        type=['jpg', 'jpeg', 'png', 'bmp', 'webp'],
        accept_multiple_files=True,
        help="Upload multiple images to process them all at once"
    )
    

    if uploaded_files and len(uploaded_files) > 0:
        st.success(f"📁 {len(uploaded_files)} images uploaded for batch processing")
        
        if st.button("🚀 Start Batch Analysis", key="batch_detect"):
            # Initialize progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            batch_results = []
            total_files = len(uploaded_files)
            
            for idx, uploaded_file in enumerate(uploaded_files):
                # Update progress
                progress = (idx + 1) / total_files
                progress_bar.progress(progress)
                status_text.text(f"Processing image {idx + 1}/{total_files}: {uploaded_file.name}")
                
                # Save file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name
                
                try:
                    # Run detection
                    result_image, detections = detector.detect_objects(tmp_file_path)
                    
                    batch_results.append({
                        'filename': uploaded_file.name,
                        'detections': detections,
                        'result_image': result_image,
                        'status': 'success'
                    })
                    
                except Exception as e:
                    batch_results.append({
                        'filename': uploaded_file.name,
                        'detections': [],
                        'result_image': None,
                        'status': 'error',
                        'error': str(e)
                    })
                
                finally:
                    try:
                        os.unlink(tmp_file_path)
                    except:
                        pass
            
            # Display batch results
            status_text.text("✅ Batch processing completed!")
            
            # Summary statistics
            successful_detections = sum(1 for r in batch_results if r['status'] == 'success')
            total_objects = sum(len(r['detections']) for r in batch_results if r['status'] == 'success')
            
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-number">{total_files}</div>
                    <div class="stat-label">Total Images</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-number">{successful_detections}</div>
                    <div class="stat-label">Successful</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-number">{total_objects}</div>
                    <div class="stat-label">Objects Found</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                success_rate = (successful_detections / total_files) * 100
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-number">{success_rate:.1f}%</div>
                    <div class="stat-label">Success Rate</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Display individual results
            st.markdown("### 📋 Individual Results")
            
            for result in batch_results:
                with st.expander(f"📄 {result['filename']} - {result['status'].title()}"):
                    if result['status'] == 'success':
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("**Detection Results:**")
                            if result['result_image'] is not None:
                                st.image(result['result_image'], caption=f"Results for {result['filename']}", use_column_width=True)
                        
                        with col2:
                            st.markdown("**Objects Detected:**")
                            if result['detections']:
                                for detection in result['detections']:
                                    st.write(f"• {detection['class']}: {detection['confidence']:.2%}")
                            else:
                                st.write("No objects detected")
                    else:
                        st.error(f"❌ Error processing {result['filename']}: {result.get('error', 'Unknown error')}")
    
                        st.markdown('</div>', unsafe_allow_html=True)

def show_enhanced_detection_summary(detections):
    """Enhanced detection summary with modern styling"""
    if not detections:
        st.markdown('<div class="detection-info">', unsafe_allow_html=True)
        st.info("🔍 No objects detected in this image")
        st.markdown('</div>', unsafe_allow_html=True)
        return
    
    st.markdown('<div class="detection-info">', unsafe_allow_html=True)
    st.markdown("### 🎯 Detection Summary")
    
    # Object count by class
    class_counts = {}
    for detection in detections:
        class_name = detection['class']
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    # Display statistics
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📊 Object Statistics:**")
        st.metric("Total Objects", len(detections))
        st.metric("Unique Classes", len(class_counts))
        
        # Average confidence
        avg_confidence = sum(d['confidence'] for d in detections) / len(detections)
        st.metric("Average Confidence", f"{avg_confidence:.1%}")
    
    with col2:
        st.markdown("**🏷️ Detected Classes:**")
        for class_name, count in sorted(class_counts.items()):
            st.write(f"• **{class_name}**: {count} object{'s' if count > 1 else ''}")
    
    # Detailed detection list
    with st.expander("🔍 Detailed Detection List"):
        for i, detection in enumerate(detections, 1):
            st.write(f"{i}. **{detection['class']}** - Confidence: {detection['confidence']:.2%}")
            if 'bbox' in detection:
                bbox = detection['bbox']
                st.write(f"   📍 Position: ({bbox[0]:.0f}, {bbox[1]:.0f}) - Size: {bbox[2]:.0f}×{bbox[3]:.0f}")
    
    st.markdown('</div>', unsafe_allow_html=True)



def show_about_and_developer():
    """Enhanced about section with developer information"""
    
    st.markdown('<div class="glass-container">', unsafe_allow_html=True)
    
    # Features section
    st.markdown("## 🚀 Features")
    
    features = [
        {
            "icon": "🤖",
            "title": "Advanced AI Detection",
            "description": "Powered by YOLOv8 - state-of-the-art object detection"
        },
        {
            "icon": "⚡",
            "title": "Real-time Processing",
            "description": "Lightning-fast detection with optimized performance"
        },
        {
            "icon": "📱",
            "title": "Multi-source Input",
            "description": "Upload images, use camera, or batch process multiple files"
        },
        {
            "icon": "🎨",
            "title": "Modern Interface",
            "description": "Beautiful, responsive design with smooth animations"
        },
        {
            "icon": "🔧",
            "title": "Configurable Models",
            "description": "Choose from multiple YOLO models for different needs"
        },
        {
            "icon": "📊",
            "title": "Detailed Analytics",
            "description": "Comprehensive detection statistics and insights"
        }
    ]
    
    st.markdown('<div class="feature-grid">', unsafe_allow_html=True)
    for feature in features:
        st.markdown(f"""
        <div class="feature-card">
            <div class="feature-icon">{feature['icon']}</div>
            <h3>{feature['title']}</h3>
            <p>{feature['description']}</p>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Technical specifications
    st.markdown("## ⚙️ Technical Specifications")
    
    tech_specs = {
        "🧠 AI Framework": "YOLOv8 (Ultralytics)",
        "🖼️ Supported Formats": "JPG, PNG, BMP, WebP",
        "⚡ Processing Speed": "Real-time detection",
        "🎯 Detection Classes": "80+ COCO dataset classes",
        "🔧 Backend": "Python + OpenCV",
        "🌐 Interface": "Streamlit Web App"
    }
    
    col1, col2 = st.columns(2)
    for i, (key, value) in enumerate(tech_specs.items()):
        col = col1 if i % 2 == 0 else col2
        with col:
            st.markdown(f"""
            <div class="tech-spec-item">
                <strong>{key}</strong>: {value}
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Developer section
    st.markdown("""
    <div class="developer-section">
        <h2>👨‍💻 Developer Information</h2>
        <h3>AI Object Detection Studio</h3>
        <p>Created with ❤️ by <strong>Deepak Singh</strong> using modern web technologies and cutting-edge AI</p>
        
        <div class="social-links">
            <a href="https://your-portfolio-url.com" class="social-link" target="_blank">
                🌐 Portfolio
            </a>
            <a href="https://linkedin.com/in/your-profile" class="social-link" target="_blank">
                💼 LinkedIn
            </a>
            <a href="https://github.com/your-username" class="social-link" target="_blank">
                🐙 GitHub
            </a>
        </div>
        
        <div class="tech-info">
            <p><strong>🔧 Tech Stack:</strong> Python • Streamlit • YOLOv8 • OpenCV • PIL</p>
            <p><strong>📅 Version:</strong> 2.0.0 • <strong>📱 Platform:</strong> Web Application</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Usage instructions
    st.markdown('<div class="glass-container">', unsafe_allow_html=True)
    st.markdown("## 📖 How to Use")
    
    instructions = [
        "1. **📁 Upload Tab**: Select and upload an image file for detection",
        "2. **📷 Camera Tab**: Use your device camera for real-time detection",
        "3. **🎯 Batch Tab**: Process multiple images simultaneously",
        "4. **⚙️ Settings**: Adjust model and confidence threshold in sidebar",
        "5. **📊 Results**: View detected objects with confidence scores and statistics"
    ]
    
    for instruction in instructions:
        st.markdown(instruction)
    
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()