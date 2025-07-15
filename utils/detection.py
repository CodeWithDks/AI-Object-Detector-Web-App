# utils/detection.py
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageEnhance, ImageFilter
import streamlit as st
import os
import tempfile
import logging
from typing import Dict, List, Tuple, Optional, Any
import torch
from collections import defaultdict
import json
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@st.cache_resource
def load_yolo_model(model_name="yolov8s.pt"):
    """Load YOLO model with caching and error handling"""
    try:
        # Create models directory if it doesn't exist
        os.makedirs("models", exist_ok=True)
        
        # Check if CUDA is available
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Load the model
        model = YOLO(model_name)
        
        # Move model to appropriate device
        if device == 'cuda':
            model.to(device)
        
        st.success(f"✅ Model {model_name} loaded successfully on {device.upper()}!")
        logger.info(f"Model {model_name} loaded on {device}")
        return model
    except Exception as e:
        st.error(f"❌ Error loading model {model_name}: {str(e)}")
        logger.error(f"Error loading model {model_name}: {str(e)}")
        return None

class AdvancedObjectDetector:
    def __init__(self):
        self.model = None
        self.model_name = "yolov8s.pt"
        self.confidence = 0.15
        self.iou_threshold = 0.45
        self.max_detections = 1000
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Advanced detection parameters
        self.use_ensemble = True
        self.use_tta = True  # Test Time Augmentation
        self.use_advanced_nms = True
        self.multi_scale_inference = True
        self.class_specific_confidence = {}
        
        # Image preprocessing parameters
        self.enhance_image = True
        self.denoise_image = True
        self.adaptive_preprocessing = True
        
        # Performance tracking
        self.detection_stats = defaultdict(int)
        
        self.load_model()
        self._initialize_class_weights()
    
    def load_model(self):
        """Load YOLO model with enhanced error handling"""
        try:
            self.model = load_yolo_model(self.model_name)
            if self.model is None:
                raise ValueError(f"Failed to load model {self.model_name}")
        except Exception as e:
            st.error(f"Error in load_model: {str(e)}")
            logger.error(f"Error in load_model: {str(e)}")
            self.model = None
    
    def _initialize_class_weights(self):
        """Initialize class-specific confidence weights"""
        # Some classes are typically harder to detect accurately
        self.class_specific_confidence = {
            'person': 0.2,
            'car': 0.25,
            'bicycle': 0.15,
            'motorcycle': 0.15,
            'airplane': 0.3,
            'bus': 0.3,
            'train': 0.3,
            'truck': 0.3,
            'boat': 0.2,
            'traffic light': 0.1,
            'stop sign': 0.2,
            'cell phone': 0.1,
            'laptop': 0.2,
            'mouse': 0.1,
            'remote': 0.1,
            'keyboard': 0.15,
            'book': 0.1,
            'clock': 0.15,
            'scissors': 0.1,
            'teddy bear': 0.15,
            'hair drier': 0.1,
            'toothbrush': 0.1
        }
    
    def update_model(self, model_name):
        """Update the model with validation"""
        if model_name != self.model_name:
            self.model_name = model_name
            try:
                old_model = self.model
                self.model = load_yolo_model(model_name)
                if self.model is None:
                    # Rollback to previous model
                    self.model = old_model
                    self.model_name = "yolov8s.pt" if old_model is None else self.model_name
                    raise ValueError(f"Failed to load model {model_name}")
            except Exception as e:
                st.error(f"Error updating model: {str(e)}")
                logger.error(f"Error updating model: {str(e)}")
    
    def set_confidence(self, confidence):
        """Set confidence threshold with validation"""
        self.confidence = max(0.01, min(1.0, confidence))
    
    def set_iou_threshold(self, iou_threshold):
        """Set IoU threshold with validation"""
        self.iou_threshold = max(0.01, min(1.0, iou_threshold))
    
    def adaptive_image_preprocessing(self, image: np.ndarray) -> np.ndarray:
        """
        Apply adaptive preprocessing based on image characteristics
        
        Args:
            image (np.ndarray): Input image
            
        Returns:
            np.ndarray: Preprocessed image
        """
        # Analyze image characteristics
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Calculate image statistics
        mean_brightness = np.mean(gray)
        std_brightness = np.std(gray)
        
        # Convert to PIL for enhancement
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Adaptive brightness adjustment
        if mean_brightness < 80:  # Dark image
            enhancer = ImageEnhance.Brightness(pil_image)
            pil_image = enhancer.enhance(1.3)
        elif mean_brightness > 200:  # Bright image
            enhancer = ImageEnhance.Brightness(pil_image)
            pil_image = enhancer.enhance(0.9)
        
        # Adaptive contrast adjustment
        if std_brightness < 30:  # Low contrast
            enhancer = ImageEnhance.Contrast(pil_image)
            pil_image = enhancer.enhance(1.4)
        elif std_brightness > 80:  # High contrast
            enhancer = ImageEnhance.Contrast(pil_image)
            pil_image = enhancer.enhance(0.9)
        
        # Sharpness enhancement
        enhancer = ImageEnhance.Sharpness(pil_image)
        pil_image = enhancer.enhance(1.15)
        
        # Color enhancement for better object distinction
        enhancer = ImageEnhance.Color(pil_image)
        pil_image = enhancer.enhance(1.1)
        
        # Apply denoising if requested
        if self.denoise_image:
            pil_image = pil_image.filter(ImageFilter.MedianFilter(size=3))
        
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    
    def preprocess_image(self, image_path: str) -> str:
        """
        Advanced image preprocessing for better detection
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            str: Path to preprocessed image
        """
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            return image_path
        
        if not self.enhance_image:
            return image_path
        
        # Apply adaptive preprocessing
        enhanced_image = self.adaptive_image_preprocessing(image)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        lab = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        enhanced_image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # Apply bilateral filter for noise reduction while preserving edges
        enhanced_image = cv2.bilateralFilter(enhanced_image, 9, 75, 75)
        
        # Save enhanced image temporarily
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
            cv2.imwrite(tmp_file.name, enhanced_image)
            return tmp_file.name
    
    def multi_scale_detection(self, image_path: str) -> List[Dict]:
        """
        Perform multi-scale detection for better accuracy
        
        Args:
            image_path (str): Path to the image
            
        Returns:
            List[Dict]: Combined detection results
        """
        all_detections = []
        
        # Different image sizes for multi-scale detection
        scales = [640, 800, 1024] if self.multi_scale_inference else [640]
        
        for scale in scales:
            try:
                results = self.model.predict(
                    source=image_path,
                    conf=self.confidence * 0.8,  # Slightly lower confidence for multi-scale
                    iou=self.iou_threshold,
                    max_det=self.max_detections,
                    save=False,
                    verbose=False,
                    augment=self.use_tta,
                    agnostic_nms=False,
                    half=False,
                    imgsz=scale,
                    device=self.device
                )
                
                for result in results:
                    if result.boxes is not None and len(result.boxes) > 0:
                        boxes = result.boxes.xyxy.cpu().numpy()
                        classes = result.boxes.cls.cpu().numpy()
                        confidences = result.boxes.conf.cpu().numpy()
                        
                        for box, cls, conf in zip(boxes, classes, confidences):
                            class_name = self.model.names[int(cls)]
                            
                            # Apply class-specific confidence threshold
                            min_conf = self.class_specific_confidence.get(class_name, self.confidence)
                            
                            if conf >= min_conf:
                                all_detections.append({
                                    'bbox': box,
                                    'class': class_name,
                                    'class_id': int(cls),
                                    'confidence': float(conf),
                                    'scale': scale
                                })
                                
            except Exception as e:
                logger.warning(f"Multi-scale detection failed for scale {scale}: {str(e)}")
                continue
        
        return all_detections
    
    def advanced_nms(self, detections: List[Dict]) -> List[Dict]:
        """
        Apply advanced Non-Maximum Suppression
        
        Args:
            detections (List[Dict]): List of detections
            
        Returns:
            List[Dict]: Filtered detections
        """
        if not detections:
            return []
        
        # Group detections by class for class-specific NMS
        class_detections = defaultdict(list)
        for detection in detections:
            class_detections[detection['class']].append(detection)
        
        final_detections = []
        
        for class_name, class_dets in class_detections.items():
            if not class_dets:
                continue
            
            # Convert to format expected by cv2.dnn.NMSBoxes
            boxes = [det['bbox'] for det in class_dets]
            scores = [det['confidence'] for det in class_dets]
            
            # Apply NMS
            indices = cv2.dnn.NMSBoxes(
                boxes,
                scores,
                self.confidence,
                self.iou_threshold
            )
            
            if len(indices) > 0:
                indices = indices.flatten()
                for i in indices:
                    final_detections.append(class_dets[i])
        
        return final_detections
    
    def detect_objects(self, image_path: str) -> Tuple[Image.Image, List[Dict]]:
        """
        Enhanced object detection with multiple improvements
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            Tuple[Image.Image, List[Dict]]: Annotated image and detections
        """
        if self.model is None:
            raise ValueError("Model not loaded. Please check your internet connection and try again.")
        
        # Verify image exists
        if not os.path.exists(image_path):
            raise ValueError(f"Image file not found: {image_path}")
        
        # Read the original image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image from {image_path}")
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Preprocess image
        enhanced_image_path = self.preprocess_image(image_path)
        
        try:
            # Perform multi-scale detection
            all_detections = self.multi_scale_detection(enhanced_image_path)
            
            # Apply advanced NMS
            if self.use_advanced_nms:
                filtered_detections = self.advanced_nms(all_detections)
            else:
                filtered_detections = all_detections
            
            # Sort by confidence
            filtered_detections.sort(key=lambda x: x['confidence'], reverse=True)
            
            # Create annotated image
            annotated_image = self.create_annotated_image(image_rgb, filtered_detections)
            
            # Format detections for output
            output_detections = []
            for detection in filtered_detections:
                x1, y1, x2, y2 = detection['bbox'].astype(int)
                output_detections.append({
                    'class': detection['class'],
                    'confidence': detection['confidence'],
                    'bbox': [int(x1), int(y1), int(x2), int(y2)]
                })
            
            # Update detection statistics
            self.detection_stats['total_detections'] += len(output_detections)
            for det in output_detections:
                self.detection_stats[det['class']] += 1
            
            # Convert to PIL Image
            pil_image = Image.fromarray(annotated_image)
            
            return pil_image, output_detections
            
        except Exception as e:
            logger.error(f"Error during detection: {str(e)}")
            raise ValueError(f"Error during model inference: {str(e)}")
        finally:
            # Clean up temporary file
            if enhanced_image_path != image_path and os.path.exists(enhanced_image_path):
                os.unlink(enhanced_image_path)
    
    def create_annotated_image(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """
        Create annotated image with enhanced visualization
        
        Args:
            image (np.ndarray): Original image
            detections (List[Dict]): List of detections
            
        Returns:
            np.ndarray: Annotated image
        """
        annotated_image = image.copy()
        
        for detection in detections:
            bbox = detection['bbox']
            class_name = detection['class']
            confidence = detection['confidence']
            class_id = detection['class_id']
            
            x1, y1, x2, y2 = bbox.astype(int)
            
            # Get color for class
            color = self.get_enhanced_color_for_class(class_id)
            
            # Draw bounding box with gradient effect
            thickness = max(2, min(4, int((x2 - x1) / 100)))
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, thickness)
            
            # Create label with confidence
            label = f"{class_name} {confidence:.2f}"
            
            # Calculate label size
            font_scale = min(0.7, max(0.4, (x2 - x1) / 200))
            font_thickness = max(1, int(font_scale * 2))
            
            (label_width, label_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
            )
            
            # Draw label background with rounded corners effect
            label_y = max(y1 - label_height - 10, 0)
            cv2.rectangle(
                annotated_image,
                (x1, label_y),
                (x1 + label_width + 10, y1),
                color,
                -1
            )
            
            # Add semi-transparent overlay
            overlay = annotated_image.copy()
            cv2.rectangle(
                overlay,
                (x1, label_y),
                (x1 + label_width + 10, y1),
                color,
                -1
            )
            cv2.addWeighted(overlay, 0.7, annotated_image, 0.3, 0, annotated_image)
            
            # Draw text
            cv2.putText(
                annotated_image,
                label,
                (x1 + 5, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                font_thickness
            )
            
            # Add confidence bar
            if confidence > 0.7:
                bar_color = (0, 255, 0)  # Green for high confidence
            elif confidence > 0.5:
                bar_color = (0, 255, 255)  # Yellow for medium confidence
            else:
                bar_color = (0, 0, 255)  # Red for low confidence
            
            bar_width = int((x2 - x1) * confidence)
            cv2.rectangle(
                annotated_image,
                (x1, y2 - 5),
                (x1 + bar_width, y2),
                bar_color,
                -1
            )
        
        return annotated_image
    
    def get_enhanced_color_for_class(self, class_id: int) -> Tuple[int, int, int]:
        """
        Get enhanced color scheme for classes
        
        Args:
            class_id (int): Class ID
            
        Returns:
            Tuple[int, int, int]: BGR color tuple
        """
        # Enhanced color palette
        colors = [
            (255, 100, 100),   # Light red
            (100, 255, 100),   # Light green
            (100, 100, 255),   # Light blue
            (255, 255, 100),   # Yellow
            (255, 100, 255),   # Magenta
            (100, 255, 255),   # Cyan
            (255, 150, 100),   # Orange
            (150, 255, 100),   # Light green
            (100, 150, 255),   # Light blue
            (255, 100, 150),   # Pink
        ]
        
        # Use consistent color for same class
        return colors[class_id % len(colors)]
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information"""
        if self.model is None:
            return {"status": "No model loaded"}
        
        return {
            'model_name': self.model_name,
            'device': self.device,
            'confidence': self.confidence,
            'iou_threshold': self.iou_threshold,
            'max_detections': self.max_detections,
            'num_classes': len(self.model.names),
            'classes': list(self.model.names.values()),
            'use_ensemble': self.use_ensemble,
            'use_tta': self.use_tta,
            'multi_scale_inference': self.multi_scale_inference,
            'detection_stats': dict(self.detection_stats)
        }
    
    def is_model_loaded(self) -> bool:
        """Check if model is loaded"""
        return self.model is not None
    
    def get_detection_summary(self, detections: List[Dict]) -> Dict[str, Any]:
        """
        Get enhanced detection summary with statistics
        
        Args:
            detections (List[Dict]): List of detections
            
        Returns:
            Dict[str, Any]: Enhanced summary
        """
        if not detections:
            return {
                "total_objects": 0,
                "unique_classes": 0,
                "object_counts": {},
                "top_detections": [],
                "confidence_stats": {},
                "detection_quality": "No detections"
            }
        
        # Count objects by class
        object_counts = defaultdict(int)
        confidence_by_class = defaultdict(list)
        
        for detection in detections:
            class_name = detection['class']
            confidence = detection['confidence']
            
            object_counts[class_name] += 1
            confidence_by_class[class_name].append(confidence)
        
        # Calculate confidence statistics
        confidence_stats = {}
        for class_name, confidences in confidence_by_class.items():
            confidence_stats[class_name] = {
                'mean': np.mean(confidences),
                'max': np.max(confidences),
                'min': np.min(confidences),
                'std': np.std(confidences)
            }
        
        # Overall detection quality assessment
        avg_confidence = np.mean([d['confidence'] for d in detections])
        if avg_confidence > 0.8:
            quality = "Excellent"
        elif avg_confidence > 0.6:
            quality = "Good"
        elif avg_confidence > 0.4:
            quality = "Fair"
        else:
            quality = "Poor"
        
        # Get top detections
        top_detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)[:5]
        
        return {
            "total_objects": len(detections),
            "unique_classes": len(object_counts),
            "object_counts": dict(object_counts),
            "top_detections": top_detections,
            "confidence_stats": confidence_stats,
            "detection_quality": quality,
            "average_confidence": avg_confidence
        }
    
    def export_detections(self, detections: List[Dict], format: str = "json") -> str:
        """
        Export detections to different formats
        
        Args:
            detections (List[Dict]): List of detections
            format (str): Export format ('json', 'csv', 'xml')
            
        Returns:
            str: Exported data as string
        """
        if format.lower() == "json":
            return json.dumps(detections, indent=2)
        elif format.lower() == "csv":
            import csv
            import io
            
            output = io.StringIO()
            writer = csv.writer(output)
            
            # Write header
            writer.writerow(['class', 'confidence', 'x1', 'y1', 'x2', 'y2'])
            
            # Write detections
            for det in detections:
                x1, y1, x2, y2 = det['bbox']
                writer.writerow([det['class'], det['confidence'], x1, y1, x2, y2])
            
            return output.getvalue()
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def reset_stats(self):
        """Reset detection statistics"""
        self.detection_stats.clear()
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        return {
            'total_detections': self.detection_stats.get('total_detections', 0),
            'class_distribution': dict(self.detection_stats),
            'device': self.device,
            'model_name': self.model_name
        }

# Backward compatibility
ObjectDetector = AdvancedObjectDetector

# utils/__init__.py
# Empty file to make utils a package