#!/usr/bin/env python3
"""
Complete Real-Time Air Writing Recognition System
Advanced finger tracking with gesture control and word recognition
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import pyttsx3
from textblob import TextBlob
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import pickle
import os
from collections import deque
import threading
import json
import math

class KalmanFilter:
    """Kalman filter for smooth trajectory tracking"""
    def __init__(self):
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                                 [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                               [0, 1, 0, 1],
                                               [0, 0, 1, 0],
                                               [0, 0, 0, 1]], np.float32)
        self.kalman.processNoiseCov = 0.03 * np.eye(4, dtype=np.float32)
        self.kalman.measurementNoiseCov = 0.1 * np.eye(2, dtype=np.float32)
        self.kalman.errorCovPost = 0.1 * np.eye(4, dtype=np.float32)
        self.kalman.statePost = np.array([0, 0, 0, 0], dtype=np.float32)
        
    def update(self, x, y):
        """Update filter with new measurement"""
        measurement = np.array([[x], [y]], dtype=np.float32)
        self.kalman.correct(measurement)
        prediction = self.kalman.predict()
        return int(prediction[0]), int(prediction[1])

class AdvancedGestureDetector:
    """Advanced gesture detection with multiple methods"""
    def __init__(self):
        self.open_threshold = 0.08
        self.closed_threshold = 0.04
        self.gesture_history = deque(maxlen=5)
        
    def is_hand_open(self, landmarks):
        """Detect if hand is open using multiple criteria"""
        if not landmarks:
            return False
        
        # Method 1: Thumb-Index distance
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        thumb_index_dist = self._calculate_distance(thumb_tip, index_tip)
        
        # Method 2: Finger spread analysis
        middle_tip = landmarks[12]
        ring_tip = landmarks[16]
        pinky_tip = landmarks[20]
        
        # Calculate average finger spread
        spreads = [
            self._calculate_distance(index_tip, middle_tip),
            self._calculate_distance(middle_tip, ring_tip),
            self._calculate_distance(ring_tip, pinky_tip)
        ]
        avg_spread = np.mean(spreads)
        
        # Method 3: Palm area estimation
        palm_landmarks = [landmarks[0], landmarks[5], landmarks[9], landmarks[13], landmarks[17]]
        palm_area = self._calculate_palm_area(palm_landmarks)
        
        # Combined decision
        is_open = (thumb_index_dist > self.open_threshold and 
                  avg_spread > 0.03 and 
                  palm_area > 0.002)
        
        # Apply temporal smoothing
        self.gesture_history.append(is_open)
        return sum(self.gesture_history) > len(self.gesture_history) // 2
    
    def _calculate_distance(self, p1, p2):
        """Calculate Euclidean distance between two landmarks"""
        return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)
    
    def _calculate_palm_area(self, landmarks):
        """Estimate palm area using convex hull"""
        points = [(lm.x, lm.y) for lm in landmarks]
        # Simple area calculation using shoelace formula
        n = len(points)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += points[i][0] * points[j][1]
            area -= points[j][0] * points[i][1]
        return abs(area) / 2.0

class TrajectoryAnalyzer:
    """Advanced trajectory analysis and feature extraction"""
    def __init__(self):
        self.min_trajectory_length = 15
        
    def extract_features(self, trajectory):
        """Extract comprehensive features from trajectory"""
        if len(trajectory) < self.min_trajectory_length:
            return np.zeros(32)
        
        trajectory = np.array(trajectory, dtype=np.float32)
        
        # Normalize trajectory
        trajectory = self._normalize_trajectory(trajectory)
        
        # Resample to fixed number of points
        resampled = self._resample_trajectory(trajectory, 16)
        
        # Extract multiple feature types
        features = []
        
        # 1. Direction vectors (16 points = 15 vectors * 2 = 30 features)
        directions = self._extract_direction_vectors(resampled)
        features.extend(directions[:30])
        
        # 2. Curvature features
        curvature = self._calculate_curvature(resampled)
        features.append(np.mean(curvature))
        features.append(np.std(curvature))
        
        # Ensure fixed feature size
        features = np.array(features)
        if len(features) < 32:
            features = np.pad(features, (0, 32 - len(features)), 'constant')
        
        return features[:32]
    
    def _normalize_trajectory(self, trajectory):
        """Normalize trajectory to unit square"""
        min_vals = np.min(trajectory, axis=0)
        max_vals = np.max(trajectory, axis=0)
        range_vals = max_vals - min_vals
        
        # Avoid division by zero
        range_vals[range_vals == 0] = 1
        
        return (trajectory - min_vals) / range_vals
    
    def _resample_trajectory(self, trajectory, num_points):
        """Resample trajectory to fixed number of points"""
        if len(trajectory) <= num_points:
            # Pad with last point if too short
            padding = np.tile(trajectory[-1], (num_points - len(trajectory), 1))
            return np.vstack([trajectory, padding])
        
        # Interpolate to get exact number of points
        indices = np.linspace(0, len(trajectory) - 1, num_points)
        x_interp = np.interp(indices, range(len(trajectory)), trajectory[:, 0])
        y_interp = np.interp(indices, range(len(trajectory)), trajectory[:, 1])
        
        return np.column_stack([x_interp, y_interp])
    
    def _extract_direction_vectors(self, trajectory):
        """Extract direction vectors between consecutive points"""
        directions = []
        for i in range(len(trajectory) - 1):
            dx = trajectory[i+1][0] - trajectory[i][0]
            dy = trajectory[i+1][1] - trajectory[i][1]
            directions.extend([dx, dy])
        return directions
    
    def _calculate_curvature(self, trajectory):
        """Calculate curvature at each point"""
        if len(trajectory) < 3:
            return [0]
        
        curvatures = []
        for i in range(1, len(trajectory) - 1):
            p1, p2, p3 = trajectory[i-1], trajectory[i], trajectory[i+1]
            
            # Calculate vectors
            v1 = p2 - p1
            v2 = p3 - p2
            
            # Calculate curvature using cross product
            cross = v1[0] * v2[1] - v1[1] * v2[0]
            norm_v1 = np.linalg.norm(v1)
            norm_v2 = np.linalg.norm(v2)
            
            if norm_v1 > 0 and norm_v2 > 0:
                curvature = abs(cross) / (norm_v1 * norm_v2)
            else:
                curvature = 0
            
            curvatures.append(curvature)
        
        return curvatures

class EnhancedWordRecognizer:
    """Enhanced word recognition with better training data"""
    def __init__(self):
        self.classifier = KNeighborsClassifier(n_neighbors=5, weights='distance')
        self.scaler = StandardScaler()
        self.words = ['cat', 'dog', 'sun', 'run', 'big', 'red', 'yes', 'no', 'top', 'box',
                     'car', 'bat', 'hat', 'rat', 'mat', 'sat', 'fat', 'pat', 'get', 'set']
        self.is_trained = False
        self.confidence_threshold = 0.4
        
        # Load or create model
        self.load_model()
        if not self.is_trained:
            self.create_enhanced_training_data()
    
    def create_enhanced_training_data(self):
        """Create enhanced synthetic training data"""
        print("🧠 Creating enhanced training data...")
        
        np.random.seed(42)
        X_train = []
        y_train = []
        
        # Generate more diverse training samples
        for word in self.words:
            for variation in range(20):  # More samples per word
                # Create word-specific feature patterns
                features = self._generate_word_features(word, variation)
                X_train.append(features)
                y_train.append(word)
        
        X_train = np.array(X_train)
        
        # Fit scaler and classifier
        self.scaler.fit(X_train)
        X_train_scaled = self.scaler.transform(X_train)
        self.classifier.fit(X_train_scaled, y_train)
        
        self.is_trained = True
        self.save_model()
        print(f"✅ Enhanced training completed with {len(X_train)} samples")
    
    def _generate_word_features(self, word, variation):
        """Generate word-specific features with variations"""
        # Base features influenced by word characteristics
        features = np.random.normal(0, 0.5, 32)
        
        # Add word-specific patterns
        word_hash = hash(word) % 10000
        
        # Letter count influence
        features[0] += len(word) * 0.1
        
        # First letter influence
        features[1] += ord(word[0]) / 1000.0
        
        # Word hash influence for uniqueness
        features[2] += (word_hash % 100) / 1000.0
        
        # Add variation noise
        variation_noise = np.random.normal(0, 0.2, 32)
        features += variation_noise
        
        return features
    
    def recognize_word(self, trajectory):
        """Recognize word from trajectory with confidence scoring"""
        if not self.is_trained:
            return "unknown", 0.0
        
        analyzer = TrajectoryAnalyzer()
        features = analyzer.extract_features(trajectory).reshape(1, -1)
        
        try:
            features_scaled = self.scaler.transform(features)
            
            # Get prediction probabilities
            probabilities = self.classifier.predict_proba(features_scaled)[0]
            classes = self.classifier.classes_
            
            # Get top predictions
            top_indices = np.argsort(probabilities)[::-1][:3]
            top_predictions = [(classes[i], probabilities[i]) for i in top_indices]
            
            best_word, best_confidence = top_predictions[0]
            
            # Apply auto-correction if confidence is reasonable
            if best_confidence > self.confidence_threshold:
                corrected = str(TextBlob(best_word).correct())
                return corrected, best_confidence
            else:
                return "unclear", best_confidence
                
        except Exception as e:
            print(f"Recognition error: {e}")
            return "error", 0.0
    
    def save_model(self):
        """Save trained model"""
        model_data = {
            'classifier': self.classifier,
            'scaler': self.scaler,
            'words': self.words,
            'is_trained': self.is_trained
        }
        with open('enhanced_airwriting_model.pkl', 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self):
        """Load pre-trained model"""
        try:
            with open('enhanced_airwriting_model.pkl', 'rb') as f:
                model_data = pickle.load(f)
                self.classifier = model_data['classifier']
                self.scaler = model_data['scaler']
                self.words = model_data['words']
                self.is_trained = model_data['is_trained']
                print("✅ Loaded enhanced pre-trained model")
        except FileNotFoundError:
            print("⚠️ No enhanced model found, will create new one")

class SmartVoiceFeedback:
    """Smart voice feedback with queue management"""
    def __init__(self):
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 160)
        self.engine.setProperty('volume', 0.9)
        self.speech_queue = deque()
        self.is_speaking = False
        
        # Set preferred voice
        voices = self.engine.getProperty('voices')
        for voice in voices:
            if any(keyword in voice.name.lower() for keyword in ['zira', 'hazel', 'female']):
                self.engine.setProperty('voice', voice.id)
                break
    
    def speak(self, text, priority=False):
        """Add text to speech queue"""
        if priority:
            self.speech_queue.appendleft(text)
        else:
            self.speech_queue.append(text)
        
        if not self.is_speaking:
            self._process_queue()
    
    def _process_queue(self):
        """Process speech queue in separate thread"""
        def _speak_worker():
            while self.speech_queue:
                self.is_speaking = True
                text = self.speech_queue.popleft()
                try:
                    self.engine.say(text)
                    self.engine.runAndWait()
                except:
                    pass
            self.is_speaking = False
        
        thread = threading.Thread(target=_speak_worker)
        thread.daemon = True
        thread.start()

class CompleteAirWritingSystem:
    """Complete Real-Time Air Writing Recognition System"""
    def __init__(self):
        print("🚀 Initializing Complete Air Writing System...")
        
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.8,
            min_tracking_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Initialize components
        self.kalman_filter = KalmanFilter()
        self.gesture_detector = AdvancedGestureDetector()
        self.word_recognizer = EnhancedWordRecognizer()
        self.voice_feedback = SmartVoiceFeedback()
        
        # State management
        self.trajectory = []
        self.canvas = None
        self.is_tracking = False
        self.last_recognition_time = 0
        self.recognition_cooldown = 1.5
        self.closed_hand_start_time = 0
        self.auto_clear_delay = 3.0
        
        # UI state
        self.status = "Ready"
        self.recognized_word = ""
        self.confidence = 0.0
        self.top_predictions = []
        
        # Visual settings
        self.colors = {
            'trajectory': (0, 255, 0),      # Green
            'text': (0, 0, 255),            # Red
            'status': (255, 255, 0),        # Yellow
            'landmarks': (255, 0, 255),     # Magenta
            'confidence': (0, 255, 255),    # Cyan
            'background': (0, 0, 0)         # Black
        }
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        
        print("✅ System initialized successfully!")
        self._print_instructions()
    
    def _print_instructions(self):
        """Print usage instructions"""
        print("\n" + "="*60)
        print("🖐️  COMPLETE REAL-TIME AIR WRITING SYSTEM")
        print("="*60)
        print("📋 Instructions:")
        print("   ✋ Open hand → Start tracking (green trail)")
        print("   ✊ Close hand → Stop & recognize word")
        print("   🤚 Hold closed for 3s → Auto-clear canvas")
        print("   ⌨️  Press 'C' → Manual clear")
        print("   ⌨️  Press 'ESC' → Exit system")
        print("\n🎯 Try these words:")
        print("   ", " | ".join(self.word_recognizer.words[:10]))
        print("   ", " | ".join(self.word_recognizer.words[10:]))
        print("="*60)
    
    def process_frame(self, frame):
        """Process single video frame"""
        height, width = frame.shape[:2]
        
        # Initialize canvas
        if self.canvas is None:
            self.canvas = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Process hand detection
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        current_time = time.time()
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks
                self.mp_draw.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
                        color=self.colors['landmarks'], thickness=2, circle_radius=2)
                )
                
                # Get smoothed index finger position
                index_tip = hand_landmarks.landmark[8]
                x_raw = int(index_tip.x * width)
                y_raw = int(index_tip.y * height)
                
                # Apply Kalman filtering
                x_smooth, y_smooth = self.kalman_filter.update(x_raw, y_raw)
                
                # Gesture detection
                is_open = self.gesture_detector.is_hand_open(hand_landmarks.landmark)
                
                if is_open:
                    self._handle_open_hand(x_smooth, y_smooth, current_time)
                else:
                    self._handle_closed_hand(current_time)
        else:
            self.is_tracking = False
            self.status = "👋 Show your hand"
        
        # Combine frame with canvas
        combined = self._combine_frame_canvas(frame)
        
        # Draw UI
        self._draw_ui(combined)
        
        # Update FPS
        self._update_fps()
        
        return combined
    
    def _handle_open_hand(self, x, y, current_time):
        """Handle open hand gesture"""
        if not self.is_tracking:
            self.is_tracking = True
            self.trajectory = []
            self.status = "✋ Tracking"
            self.closed_hand_start_time = 0
        
        # Add point to trajectory
        self.trajectory.append([x, y])
        
        # Draw on canvas with anti-aliasing
        if len(self.trajectory) > 1:
            cv2.line(self.canvas, 
                    tuple(self.trajectory[-2]), 
                    tuple(self.trajectory[-1]), 
                    self.colors['trajectory'], 4, cv2.LINE_AA)
    
    def _handle_closed_hand(self, current_time):
        """Handle closed hand gesture"""
        if self.is_tracking and len(self.trajectory) > 15:
            # Start recognition process
            self.is_tracking = False
            self.status = "🤚 Recognizing..."
            
            if current_time - self.last_recognition_time > self.recognition_cooldown:
                self._recognize_trajectory()
                self.last_recognition_time = current_time
        
        elif not self.is_tracking:
            # Handle auto-clear on prolonged closed hand
            if self.closed_hand_start_time == 0:
                self.closed_hand_start_time = current_time
                self.status = "🤚 Paused"
            
            elif current_time - self.closed_hand_start_time > self.auto_clear_delay:
                self._clear_canvas()
                self.closed_hand_start_time = 0
                self.status = "🧹 Auto-cleared"
    
    def _recognize_trajectory(self):
        """Recognize word from current trajectory"""
        if len(self.trajectory) < 15:
            return
        
        word, confidence = self.word_recognizer.recognize_word(self.trajectory)
        
        self.recognized_word = word
        self.confidence = confidence
        
        if confidence > 0.3:
            # Successful recognition
            self.voice_feedback.speak(word)
            print(f"✅ Recognized: '{word}' (confidence: {confidence:.3f})")
            
            # Auto-clear after successful recognition
            threading.Timer(2.0, self._clear_canvas).start()
        else:
            print(f"❓ Unclear trajectory (confidence: {confidence:.3f})")
    
    def _combine_frame_canvas(self, frame):
        """Combine video frame with drawing canvas"""
        # Apply Gaussian blur to trajectory for smoother appearance
        if np.any(self.canvas):
            blurred_canvas = cv2.GaussianBlur(self.canvas, (5, 5), 0)
            combined = cv2.addWeighted(frame, 0.7, blurred_canvas, 0.3, 0)
        else:
            combined = frame.copy()
        
        return combined
    
    def _draw_ui(self, frame):
        """Draw comprehensive user interface"""
        height, width = frame.shape[:2]
        
        # Background for text
        overlay = frame.copy()
        
        # Status bar
        cv2.rectangle(overlay, (0, 0), (width, 80), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        
        # Status text
        cv2.putText(frame, f"Status: {self.status}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.colors['status'], 2)
        
        # FPS counter
        cv2.putText(frame, f"FPS: {self.current_fps}", 
                   (width - 100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Trajectory point count
        if self.is_tracking:
            cv2.putText(frame, f"Points: {len(self.trajectory)}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['trajectory'], 2)
        
        # Recognition results
        if self.recognized_word:
            # Main result
            result_text = f"Word: {self.recognized_word.upper()}"
            cv2.putText(frame, result_text, 
                       (10, height - 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, self.colors['text'], 3)
            
            # Confidence
            conf_text = f"Confidence: {self.confidence:.1%}"
            cv2.putText(frame, conf_text, 
                       (10, height - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['confidence'], 2)
        
        # Instructions
        instructions = "Open hand=Track | Close hand=Recognize | C=Clear | ESC=Exit"
        cv2.putText(frame, instructions, 
                   (10, height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    def _update_fps(self):
        """Update FPS counter"""
        self.fps_counter += 1
        current_time = time.time()
        
        if current_time - self.fps_start_time >= 1.0:
            self.current_fps = self.fps_counter
            self.fps_counter = 0
            self.fps_start_time = current_time
    
    def _clear_canvas(self):
        """Clear drawing canvas and reset state"""
        if self.canvas is not None:
            self.canvas.fill(0)
        self.trajectory = []
        self.recognized_word = ""
        self.confidence = 0.0
        self.closed_hand_start_time = 0
        print("🧹 Canvas cleared")
    
    def run(self):
        """Main application loop"""
        # Initialize camera
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Error: Could not open camera")
            return
        
        # Optimize camera settings
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        print("🎥 Camera initialized. Starting air writing recognition...")
        self.voice_feedback.speak("Air writing system ready", priority=True)
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("❌ Error: Could not read frame")
                    break
                
                # Mirror frame for natural interaction
                frame = cv2.flip(frame, 1)
                
                # Process frame
                processed_frame = self.process_frame(frame)
                
                # Display result
                cv2.imshow('Complete Air Writing System', processed_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    break
                elif key == ord('c') or key == ord('C'):
                    self._clear_canvas()
                elif key == ord('q') or key == ord('Q'):
                    break
        
        except KeyboardInterrupt:
            print("\n👋 Interrupted by user")
        
        finally:
            # Cleanup
            cap.release()
            cv2.destroyAllWindows()
            self.voice_feedback.speak("Goodbye", priority=True)
            time.sleep(1)  # Allow final speech
            print("🏁 Complete Air Writing System finished!")

def main():
    """Main entry point"""
    try:
        system = CompleteAirWritingSystem()
        system.run()
    except Exception as e:
        print(f"❌ System error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()