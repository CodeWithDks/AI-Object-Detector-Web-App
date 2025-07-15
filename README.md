# 🔍 Object Detection Web App

A modern, user-friendly web application for real-time object detection using YOLOv8 and Streamlit.

## ✨ Features

- **📁 Image Upload**: Upload images in various formats (JPG, PNG, BMP, WebP)
- **📷 Camera Integration**: Take photos using your device's camera
- **🎥 Real-time Video**: Live object detection from webcam
- **⚙️ Customizable Settings**: Adjust confidence threshold and model selection
- **🎯 Multiple Models**: Choose from YOLOv8n, YOLOv8s, YOLOv8m, YOLOv8l
- **📊 Detailed Results**: View detection summaries and bounding box information

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd object_detection_app

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Create Directory Structure

```bash
mkdir -p utils models uploads
touch utils/__init__.py
```

### 3. Run the Application

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## 📁 Project Structure

```
object_detection_app/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── utils/
│   ├── __init__.py
│   └── detection.py      # Object detection utilities
├── models/               # YOLO models (auto-downloaded)
├── uploads/             # Temporary image storage
└── README.md           # This file
```

## 🎯 Usage

### Image Upload
1. Go to the "Upload Image" tab
2. Select an image file
3. Adjust confidence threshold if needed
4. Click "Detect Objects"

### Camera Detection
1. Go to the "Camera" tab
2. Choose "Take Photo" or "Real-time Video"
3. Allow camera permissions
4. For photos: Click capture, then "Detect Objects"
5. For video: Click "Start Video Detection"

### Settings
- **Model Selection**: Choose between YOLOv8n (fastest) to YOLOv8l (most accurate)
- **Confidence Threshold**: Adjust detection sensitivity (0.1 to 1.0)

## 🔧 Technical Details

### Supported Models
- **YOLOv8n**: Nano - Fastest inference, good for real-time applications
- **YOLOv8s**: Small - Balanced speed and accuracy
- **YOLOv8m**: Medium - Better accuracy, moderate speed
- **YOLOv8l**: Large - Highest accuracy, slower inference

### Object Classes
The app can detect 80 different object classes including:
- People
- Vehicles (car, truck, bus, motorcycle, bicycle)
- Animals (dog, cat, bird, horse, cow, etc.)
- Everyday objects (chair, table, laptop, phone, bottle, etc.)
- And many more!

### Performance Tips
- Use YOLOv8n for real-time video detection
- Use YOLOv8m or YOLOv8l for high-accuracy image detection
- Adjust confidence threshold based on your use case
- Ensure good lighting for better detection results

## 🛠️ Customization

### Adding New Features
1. **New Detection Models**: Add model options in the sidebar
2. **Custom Classes**: Modify detection classes in `utils/detection.py`
3. **UI Improvements**: Update styling in the CSS section of `app.py`

### Environment Variables
Create a `.env` file for configuration:
```env
DEFAULT_MODEL=yolov8n.pt
DEFAULT_CONFIDENCE=0.3
MAX_UPLOAD_SIZE=50MB
```

## 📊 API Reference

### ObjectDetector Class

```python
detector = ObjectDetector()

# Update model
detector.update_model("yolov8m.pt")

# Set confidence
detector.set_confidence(0.5)

# Detect objects
result_image, detections = detector.detect_objects("image.png")
```

### Detection Output Format
```python
detections = [
    {
        'class': 'person',
        'confidence': 0.95,
        'bbox': [x1, y1, x2, y2]
    },
    # ... more detections
]
```

## 🚨 Troubleshooting

### Common Issues

1. **Camera not working**
   - Check browser permissions
   - Ensure camera is not used by another application
   - Try refreshing the page

2. **Model loading errors**
   - Check internet connection (models are downloaded on first use)
   - Verify sufficient disk space
   - Try clearing browser cache

3. **Slow performance**
   - Use YOLOv8n for faster inference
   - Reduce image size
   - Close other applications

### Dependencies Issues
```bash
# Update pip
pip install --upgrade pip

# Reinstall requirements
pip install -r requirements.txt --upgrade
```

## 📝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Ultralytics](https://ultralytics.com/) for YOLOv8
- [Streamlit](https://streamlit.io/) for the web framework
- [OpenCV](https://opencv.org/) for computer vision utilities

## 📞 Support

If you encounter any issues or have questions:
1. Check the troubleshooting section
2. Open an issue on GitHub
3. Contact the development team

---

**Built with ❤️ using Streamlit and YOLOv8**