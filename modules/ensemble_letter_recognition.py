#!/usr/bin/env python3
"""
Ensemble Letter Recognition Module
Features:
- Multiple model ensemble for improved accuracy
- Advanced preprocessing and augmentation
- Confidence scoring and uncertainty estimation
- Real-time optimization
- Fallback recognition systems
"""

import cv2
import numpy as np
import time
import os
from collections import deque
from pathlib import Path

# Try imports with fallbacks
try:
    from tensorflow.keras.models import load_model
    import tensorflow as tf
    MODEL_AVAILABLE = True
except ImportError:
    MODEL_AVAILABLE = False
    print("⚠️  TensorFlow not available - using fallback recognition")

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    import joblib
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠️  Scikit-learn not available - using basic fallback")

class PathPreprocessor:
    """Advanced path preprocessing for letter recognition"""
    
    def __init__(self):
        self.smoothing_window = 3
        self.min_path_length = 5
        
    def smooth_path(self, path, window_size=None):
        """Apply smoothing to path using moving average"""
        if not path or len(path) < 2:
            return path
        
        window_size = window_size or self.smoothing_window
        if len(path) <= window_size:
            return path
        
        try:
            path_array = np.array(path)
            smoothed = []
            
            for i in range(len(path)):
                start_idx = max(0, i - window_size // 2)
                end_idx = min(len(path), i + window_size // 2 + 1)
                window_points = path_array[start_idx:end_idx]
                smoothed_point = np.mean(window_points, axis=0)
                smoothed.append(tuple(smoothed_point.astype(int)))
            
            return smoothed
        except Exception as e:
            print(f"⚠️ Path smoothing error: {e}")
            return path
    
    def normalize_path(self, path):
        """Normalize path to standard coordinate system"""
        if not path or len(path) < 2:
            return path
        
        try:
            path_array = np.array(path)
            
            # Get bounding box
            min_coords = np.min(path_array, axis=0)
            max_coords = np.max(path_array, axis=0)
            
            # Calculate dimensions
            width = max_coords[0] - min_coords[0]
            height = max_coords[1] - min_coords[1]
            
            if width == 0 or height == 0:
                return path
            
            # Normalize to unit square
            normalized = []
            for x, y in path:
                norm_x = (x - min_coords[0]) / width
                norm_y = (y - min_coords[1]) / height
                normalized.append((norm_x, norm_y))
            
            return normalized
        except Exception as e:
            print(f"⚠️ Path normalization error: {e}")
            return path
    
    def resample_path(self, path, target_points=50):
        """Resample path to have consistent number of points"""
        if not path or len(path) < 2:
            return path
        
        try:
            if len(path) == target_points:
                return path
            
            path_array = np.array(path)
            
            # Calculate cumulative distances
            distances = [0]
            for i in range(1, len(path_array)):
                dist = np.linalg.norm(path_array[i] - path_array[i-1])
                distances.append(distances[-1] + dist)
            
            total_distance = distances[-1]
            if total_distance == 0:
                return path
            
            # Resample at equal intervals
            resampled = []
            for i in range(target_points):
                target_dist = (i / (target_points - 1)) * total_distance
                
                # Find closest points
                idx = np.searchsorted(distances, target_dist)
                if idx == 0:
                    resampled.append(tuple(path_array[0]))
                elif idx >= len(path_array):
                    resampled.append(tuple(path_array[-1]))
                else:
                    # Interpolate between points
                    t = (target_dist - distances[idx-1]) / (distances[idx] - distances[idx-1])
                    point = path_array[idx-1] + t * (path_array[idx] - path_array[idx-1])
                    resampled.append(tuple(point.astype(int)))
            
            return resampled
        except Exception as e:
            print(f"⚠️ Path resampling error: {e}")
            return path
    
    def extract_features(self, path):
        """Extract geometric features from path"""
        if not path or len(path) < 2:
            return np.zeros(20)  # Return zero features
        
        try:
            path_array = np.array(path)
            features = []
            
            # Basic statistics
            features.extend([len(path)])  # Path length
            
            # Bounding box features
            min_coords = np.min(path_array, axis=0)
            max_coords = np.max(path_array, axis=0)
            width = max_coords[0] - min_coords[0]
            height = max_coords[1] - min_coords[1]
            aspect_ratio = width / max(height, 1)
            
            features.extend([width, height, aspect_ratio])
            
            # Path statistics
            if len(path_array) > 1:
                # Calculate distances between consecutive points
                diffs = np.diff(path_array, axis=0)
                distances = np.linalg.norm(diffs, axis=1)
                
                features.extend([
                    np.mean(distances),    # Average step size
                    np.std(distances),     # Step size variation
                    np.sum(distances),     # Total path length
                ])
                
                # Direction changes
                if len(diffs) > 1:
                    angles = []
                    for i in range(1, len(diffs)):
                        v1, v2 = diffs[i-1], diffs[i]
                        if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                            cos_angle = np.clip(cos_angle, -1, 1)
                            angle = np.arccos(cos_angle)
                            angles.append(angle)
                    
                    if angles:
                        features.extend([
                            np.mean(angles),   # Average direction change
                            np.std(angles),    # Direction change variation
                            len(angles)        # Number of direction changes
                        ])
                    else:
                        features.extend([0, 0, 0])
                else:
                    features.extend([0, 0, 0])
            else:
                features.extend([0, 0, 0, 0, 0, 0])
            
            # Curvature features
            if len(path_array) > 2:
                curvatures = []
                for i in range(1, len(path_array) - 1):
                    p1, p2, p3 = path_array[i-1], path_array[i], path_array[i+1]
                    
                    # Calculate curvature using three points
                    v1 = p2 - p1
                    v2 = p3 - p2
                    
                    if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                        cross_product = np.cross(v1, v2)
                        curvature = abs(cross_product) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                        curvatures.append(curvature)
                
                if curvatures:
                    features.extend([
                        np.mean(curvatures),  # Average curvature
                        np.max(curvatures),   # Maximum curvature
                        np.std(curvatures)    # Curvature variation
                    ])
                else:
                    features.extend([0, 0, 0])
            else:
                features.extend([0, 0, 0])
            
            # Pad or truncate to fixed size
            target_size = 20
            if len(features) < target_size:
                features.extend([0] * (target_size - len(features)))
            else:
                features = features[:target_size]
            
            return np.array(features)
            
        except Exception as e:
            print(f"⚠️ Feature extraction error: {e}")
            return np.zeros(20)

class ImageGenerator:
    """Generate images from paths for CNN models"""
    
    def __init__(self, img_size=28):
        self.img_size = img_size
        
    def create_image(self, path, enhance=True):
        """Create image from path with optional enhancement"""
        if not path or len(path) < 2:
            return np.ones((self.img_size, self.img_size), dtype=np.uint8) * 255
        
        try:
            # Create blank image
            img = np.ones((self.img_size, self.img_size), dtype=np.uint8) * 255
            
            path_array = np.array(path)
            
            # Get bounding box
            min_coords = np.min(path_array, axis=0)
            max_coords = np.max(path_array, axis=0)
            
            width = max_coords[0] - min_coords[0]
            height = max_coords[1] - min_coords[1]
            
            if width <= 0 or height <= 0:
                return img
            
            # Scale to fit image with padding
            padding = 4
            available_size = self.img_size - 2 * padding
            scale = min(available_size / width, available_size / height)
            
            # Center the path
            center_x = (min_coords[0] + max_coords[0]) / 2
            center_y = (min_coords[1] + max_coords[1]) / 2
            offset_x = self.img_size / 2 - center_x * scale
            offset_y = self.img_size / 2 - center_y * scale
            
            # Draw path
            scaled_path = []
            for x, y in path:
                new_x = int(x * scale + offset_x)
                new_y = int(y * scale + offset_y)
                new_x = np.clip(new_x, 0, self.img_size - 1)
                new_y = np.clip(new_y, 0, self.img_size - 1)
                scaled_path.append((new_x, new_y))
            
            # Draw lines with variable thickness
            for i in range(1, len(scaled_path)):
                thickness = 2
                cv2.line(img, scaled_path[i-1], scaled_path[i], (0), thickness)
            
            # Enhancement
            if enhance:
                img = self.enhance_image(img)
            
            return img
            
        except Exception as e:
            print(f"⚠️ Image creation error: {e}")
            return np.ones((self.img_size, self.img_size), dtype=np.uint8) * 255
    
    def enhance_image(self, img):
        """Apply image enhancement techniques"""
        try:
            # Apply slight blur for anti-aliasing
            img = cv2.GaussianBlur(img, (3, 3), 0.5)
            
            # Enhance contrast
            img = cv2.convertScaleAbs(img, alpha=1.1, beta=-10)
            
            # Ensure proper range
            img = np.clip(img, 0, 255).astype(np.uint8)
            
            return img
        except Exception as e:
            print(f"⚠️ Image enhancement error: {e}")
            return img
    
    def create_augmented_images(self, path, num_augmentations=3):
        """Create multiple augmented versions of the image"""
        base_img = self.create_image(path)
        augmented = [base_img]
        
        try:
            for _ in range(num_augmentations):
                # Apply random transformations
                img = base_img.copy()
                
                # Random rotation (-10 to 10 degrees)
                angle = np.random.uniform(-10, 10)
                center = (self.img_size // 2, self.img_size // 2)
                rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                img = cv2.warpAffine(img, rotation_matrix, (self.img_size, self.img_size), 
                                   borderValue=255)
                
                # Random scaling (0.9 to 1.1)
                scale = np.random.uniform(0.9, 1.1)
                scaled_size = int(self.img_size * scale)
                img = cv2.resize(img, (scaled_size, scaled_size))
                
                # Crop or pad to original size
                if scaled_size > self.img_size:
                    # Crop
                    start = (scaled_size - self.img_size) // 2
                    img = img[start:start+self.img_size, start:start+self.img_size]
                elif scaled_size < self.img_size:
                    # Pad
                    pad = (self.img_size - scaled_size) // 2
                    img = cv2.copyMakeBorder(img, pad, pad, pad, pad, 
                                           cv2.BORDER_CONSTANT, value=255)
                
                # Random noise
                noise = np.random.normal(0, 5, img.shape).astype(np.int16)
                img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
                
                augmented.append(img)
            
            return augmented
        except Exception as e:
            print(f"⚠️ Image augmentation error: {e}")
            return [base_img]

class EnsembleLetterRecognizer:
    """Ensemble letter recognizer with multiple models and approaches"""
    
    def __init__(self, model_paths=None):
        self.cnn_models = []
        self.sklearn_models = []
        self.model_weights = []
        
        # Components
        self.preprocessor = PathPreprocessor()
        self.image_generator = ImageGenerator()
        
        # Performance tracking
        self.processing_times = deque(maxlen=50)
        self.accuracy_history = deque(maxlen=100)
        
        # Load models
        self.load_models(model_paths)
        
        print(f"🧠 Ensemble recognizer initialized:")
        print(f"   • CNN models: {len(self.cnn_models)}")
        print(f"   • Sklearn models: {len(self.sklearn_models)}")
    
    def load_models(self, model_paths=None):
        """Load all available models"""
        # Load CNN models
        self.load_cnn_models(model_paths)
        
        # Load sklearn models
        self.load_sklearn_models()
        
        # Set model weights based on expected performance
        self.calculate_model_weights()
    
    def load_cnn_models(self, model_paths=None):
        """Load CNN models"""
        if not MODEL_AVAILABLE:
            return
        
        # Default model paths
        default_paths = [
            "models/letter_recognition_optimized_accurate.h5",
            "models/optimized_accurate_letter_recognition.h5",
            "models/letter_recognition.h5"
        ]
        
        paths_to_try = model_paths if model_paths else default_paths
        
        for path in paths_to_try:
            if os.path.exists(path):
                try:
                    model = load_model(path, compile=False)
                    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
                    
                    # Test model
                    test_input = np.zeros((1, 28, 28, 1))
                    _ = model.predict(test_input, verbose=0)
                    
                    self.cnn_models.append({
                        'model': model,
                        'path': path,
                        'type': 'cnn'
                    })
                    print(f"✅ Loaded CNN model: {os.path.basename(path)}")
                    
                except Exception as e:
                    print(f"⚠️  Failed to load CNN model {path}: {e}")
    
    def load_sklearn_models(self):
        """Load or create sklearn models"""
        if not SKLEARN_AVAILABLE:
            return
        
        # Try to load pre-trained sklearn models
        sklearn_model_paths = [
            "models/random_forest_letter_model.joblib",
            "models/svm_letter_model.joblib"
        ]
        
        for path in sklearn_model_paths:
            if os.path.exists(path):
                try:
                    model = joblib.load(path)
                    self.sklearn_models.append({
                        'model': model,
                        'path': path,
                        'type': 'sklearn'
                    })
                    print(f"✅ Loaded sklearn model: {os.path.basename(path)}")
                except Exception as e:
                    print(f"⚠️  Failed to load sklearn model {path}: {e}")
    
    def calculate_model_weights(self):
        """Calculate weights for ensemble voting"""
        total_models = len(self.cnn_models) + len(self.sklearn_models)
        
        if total_models == 0:
            self.model_weights = []
            return
        
        # CNN models typically perform better, give them higher weight
        cnn_weight = 0.7 / max(len(self.cnn_models), 1)
        sklearn_weight = 0.3 / max(len(self.sklearn_models), 1)
        
        self.model_weights = []
        
        # Add CNN model weights
        for _ in self.cnn_models:
            self.model_weights.append(cnn_weight)
        
        # Add sklearn model weights
        for _ in self.sklearn_models:
            self.model_weights.append(sklearn_weight)
        
        # Normalize weights
        total_weight = sum(self.model_weights)
        if total_weight > 0:
            self.model_weights = [w / total_weight for w in self.model_weights]
    
    def recognize_letter(self, path):
        """Recognize letter using ensemble approach"""
        if not path or len(path) < 3:
            return None, 0.0
        
        start_time = time.time()
        
        try:
            # Preprocess path
            smoothed_path = self.preprocessor.smooth_path(path)
            
            # Get predictions from all models
            predictions = []
            confidences = []
            
            # CNN model predictions
            if self.cnn_models:
                cnn_pred, cnn_conf = self.get_cnn_predictions(smoothed_path)
                if cnn_pred is not None:
                    predictions.extend(cnn_pred)
                    confidences.extend(cnn_conf)
            
            # Sklearn model predictions
            if self.sklearn_models:
                sklearn_pred, sklearn_conf = self.get_sklearn_predictions(smoothed_path)
                if sklearn_pred is not None:
                    predictions.extend(sklearn_pred)
                    confidences.extend(sklearn_conf)
            
            # Ensemble voting
            if predictions:
                final_letter, final_confidence = self.ensemble_vote(predictions, confidences)
            else:
                # Fallback recognition
                final_letter, final_confidence = self.fallback_recognition(smoothed_path)
            
            # Track processing time
            processing_time = (time.time() - start_time) * 1000
            self.processing_times.append(processing_time)
            
            return final_letter, final_confidence
            
        except Exception as e:
            print(f"❌ Recognition error: {e}")
            return self.fallback_recognition(path)
    
    def get_cnn_predictions(self, path):
        """Get predictions from CNN models"""
        try:
            # Create image
            img = self.image_generator.create_image(path)
            img_normalized = img.reshape(1, 28, 28, 1).astype(np.float32) / 255.0
            
            predictions = []
            confidences = []
            
            for model_info in self.cnn_models:
                try:
                    model = model_info['model']
                    pred = model.predict(img_normalized, verbose=0)[0]
                    
                    letter_idx = np.argmax(pred)
                    letter = chr(letter_idx + ord('A'))
                    confidence = pred[letter_idx]
                    
                    predictions.append(letter)
                    confidences.append(confidence)
                    
                except Exception as e:
                    print(f"⚠️ CNN model prediction error: {e}")
                    continue
            
            return predictions, confidences
            
        except Exception as e:
            print(f"⚠️ CNN predictions error: {e}")
            return None, None
    
    def get_sklearn_predictions(self, path):
        """Get predictions from sklearn models"""
        try:
            # Extract features
            features = self.preprocessor.extract_features(path)
            features = features.reshape(1, -1)
            
            predictions = []
            confidences = []
            
            for model_info in self.sklearn_models:
                try:
                    model = model_info['model']
                    
                    # Get prediction
                    pred = model.predict(features)[0]
                    
                    # Get confidence (probability if available)
                    if hasattr(model, 'predict_proba'):
                        proba = model.predict_proba(features)[0]
                        confidence = np.max(proba)
                    else:
                        confidence = 0.7  # Default confidence for non-probabilistic models
                    
                    # Convert prediction to letter
                    if isinstance(pred, (int, np.integer)):
                        letter = chr(pred + ord('A'))
                    else:
                        letter = str(pred).upper()
                    
                    predictions.append(letter)
                    confidences.append(confidence)
                    
                except Exception as e:
                    print(f"⚠️ Sklearn model prediction error: {e}")
                    continue
            
            return predictions, confidences
            
        except Exception as e:
            print(f"⚠️ Sklearn predictions error: {e}")
            return None, None
    
    def ensemble_vote(self, predictions, confidences):
        """Combine predictions using weighted voting"""
        if not predictions:
            return None, 0.0
        
        try:
            # Count votes for each letter
            letter_votes = {}
            letter_confidences = {}
            
            for i, (letter, confidence) in enumerate(zip(predictions, confidences)):
                weight = self.model_weights[i] if i < len(self.model_weights) else 1.0
                weighted_confidence = confidence * weight
                
                if letter not in letter_votes:
                    letter_votes[letter] = 0
                    letter_confidences[letter] = []
                
                letter_votes[letter] += weighted_confidence
                letter_confidences[letter].append(confidence)
            
            # Find letter with highest weighted vote
            best_letter = max(letter_votes, key=letter_votes.get)
            
            # Calculate final confidence
            final_confidence = letter_votes[best_letter]
            
            # Boost confidence if multiple models agree
            if len(letter_confidences[best_letter]) > 1:
                agreement_boost = min(0.2, len(letter_confidences[best_letter]) * 0.05)
                final_confidence += agreement_boost
            
            final_confidence = min(1.0, final_confidence)
            
            return best_letter, final_confidence
            
        except Exception as e:
            print(f"⚠️ Ensemble voting error: {e}")
            return predictions[0], confidences[0]
    
    def fallback_recognition(self, path):
        """Fallback recognition when no models are available"""
        try:
            # Simple heuristic-based recognition
            features = self.preprocessor.extract_features(path)
            
            # Very basic letter classification based on features
            aspect_ratio = features[3] if len(features) > 3 else 1.0
            path_length = features[0] if len(features) > 0 else 10
            
            # Simple rules (this is very basic and not accurate)
            if aspect_ratio > 1.5:  # Wide letters
                letters = ['H', 'M', 'W', 'N']
            elif aspect_ratio < 0.7:  # Tall letters
                letters = ['I', 'L', 'J', 'T']
            else:  # Square-ish letters
                letters = ['O', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'K', 'P', 'Q', 'R', 'S', 'U', 'V', 'X', 'Y', 'Z']
            
            # Random selection from appropriate category
            letter = np.random.choice(letters)
            confidence = np.random.uniform(0.3, 0.7)
            
            return letter, confidence
            
        except Exception as e:
            print(f"⚠️ Fallback recognition error: {e}")
            # Ultimate fallback
            letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
            letter = np.random.choice(list(letters))
            confidence = 0.5
            return letter, confidence
    
    def get_average_processing_time(self):
        """Get average processing time"""
        return np.mean(self.processing_times) if self.processing_times else 0.0
    
    def get_model_info(self):
        """Get information about loaded models"""
        info = {
            'cnn_models': len(self.cnn_models),
            'sklearn_models': len(self.sklearn_models),
            'total_models': len(self.cnn_models) + len(self.sklearn_models),
            'model_weights': self.model_weights,
            'avg_processing_time': self.get_average_processing_time()
        }
        return info

def demo_ensemble_recognition():
    """Demo function for ensemble recognition"""
    print("🚀 Starting Ensemble Letter Recognition Demo...")
    
    # Initialize recognizer
    recognizer = EnsembleLetterRecognizer()
    
    # Print model info
    info = recognizer.get_model_info()
    print(f"📊 Model Info: {info}")
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open camera")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # Tracking variables
    current_path = []
    recognized_letters = []
    
    print("📋 Instructions:")
    print("   • Draw letters in the air with your finger")
    print("   • Press SPACE to recognize current path")
    print("   • Press C to clear path")
    print("   • Press ESC to exit")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Flip for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Simple mouse tracking for demo (replace with hand tracking)
            # This is just for demonstration - in real use, integrate with hand tracker
            
            # Display current path
            if len(current_path) > 1:
                for i in range(1, len(current_path)):
                    cv2.line(frame, current_path[i-1], current_path[i], (0, 255, 0), 3)
            
            # Display recognized letters
            if recognized_letters:
                letters_text = "Letters: " + " ".join(recognized_letters[-10:])
                cv2.putText(frame, letters_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Display instructions
            cv2.putText(frame, "Draw letter, press SPACE to recognize", 
                       (10, frame.shape[0] - 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Show frame
            cv2.imshow('Ensemble Letter Recognition Demo', frame)
            
            # Handle input
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):  # SPACE - recognize
                if len(current_path) > 5:
                    letter, confidence = recognizer.recognize_letter(current_path)
                    if letter:
                        recognized_letters.append(f"{letter}({confidence:.2f})")
                        print(f"✅ Recognized: {letter} (confidence: {confidence:.3f})")
                    current_path.clear()
            elif key == ord('c') or key == ord('C'):  # Clear
                current_path.clear()
                print("🧹 Path cleared")
            elif key == 27:  # ESC
                break
    
    except KeyboardInterrupt:
        print("\n👋 Demo interrupted by user")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("✅ Demo finished")

if __name__ == "__main__":
    demo_ensemble_recognition()