#!/usr/bin/env python3
"""
Complete Real-Time Air Writing Recognition System
Detects index finger movement to write words in the air with gesture control
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

class MovingAverageFilter:
    """Simple moving average filter for smoothing coordinates"""
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.x_buffer = deque(maxlen=window_size)
        self.y_buffer = deque(maxlen=window_size)
    
    def filter(self, x, y):
        self.x_buffer.append(x)
        self.y_buffer.append(y)
        return sum(self.x_buffer) / len(self.x_buffer), sum(self.y_buffer) / len(self.y_buffer)

class GestureDetector:
    """Detects open/closed hand gestures"""
    def __init__(self, threshold=0.05):
        self.threshold = threshold
        
    def is_hand_open(self, landmarks):
        """Check if hand is open based on thumb-index distance"""
        if not landmarks:
            return False
            
        # Get thumb tip (4) and index tip (8) landmarks
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        
        # Calculate distance
        distance = np.sqrt((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2)
        
        return distance > self.threshold

class TrajectoryProcessor:
    """Processes and extracts features from finger trajectories"""
    def __init__(self):
        self.scaler = StandardScaler()
        
    def extract_features(self, trajectory):
        """Extract features from trajectory for word recognition"""
        if len(trajectory) < 3:
            return np.zeros(20)  # Return zero vector for short trajectories
            
        trajectory = np.array(trajectory)
        
        # Resample trajectory to fixed number of points
        resampled = self.resample_trajectory(trajectory, 10)
        
        # Extract direction vectors
        directions = []
        for i in range(len(resampled) - 1):
            dx = resampled[i+1][0] - resampled[i][0]
            dy = resampled[i+1][1] - resampled[i][1]
            directions.extend([dx, dy])
        
        # Pad or truncate to fixed size
        features = np.array(directions[:18])  # 9 direction vectors * 2
        if len(features) < 18:
            features = np.pad(features, (0, 18 - len(features)), 'constant')
            
        # Add trajectory statistics
        stats = [
            np.mean(trajectory[:, 0]),  # Mean X
            np.mean(trajectory[:, 1])   # Mean Y
        ]
        
        return np.concatenate([features, stats])
    
    def resample_trajectory(self, trajectory, num_points):
        """Resample trajectory to fixed number of points"""
        if len(trajectory) <= num_points:
            return trajectory
            
        indices = np.linspace(0, len(trajectory) - 1, num_points, dtype=int)
        return trajectory[indices]

class WordRecognizer:
    """Recognizes words from trajectory features"""
    def __init__(self):
        self.classifier = KNeighborsClassifier(n_neighbors=3)
        self.scaler = StandardScaler()
        self.words = ['cat', 'dog', 'sun', 'run', 'big', 'red', 'yes', 'no', 'top', 'box']
        self.is_trained = False
        
        # Try to load pre-trained model
        self.load_model()
        
        # If no model exists, create basic training data
        if not self.is_trained:
            self.create_basic_training_data()
    
    def create_basic_training_data(self):
        """Create basic training data for demonstration"""
        print("🧠 Creating basic training data...")
        
        # Generate synthetic training data for demonstration
        np.random.seed(42)
        X_train = []
        y_train = []
        
        for word in self.words:
            # Generate 10 synthetic samples per word
            for _ in range(10):
                # Create random but consistent features for each word
                base_features = np.random.normal(0, 1, 20)
                # Add word-specific bias
                word_bias = hash(word) % 1000 / 1000.0
                base_features[0] += word_bias
                X_train.append(base_features)
                y_train.append(word)
        
        X_train = np.array(X_train)
        self.scaler.fit(X_train)
        X_train_scaled = self.scaler.transform(X_train)
        
        self.classifier.fit(X_train_scaled, y_train)
        self.is_trained = True
        
        # Save the model
        self.save_model()
        print("✅ Basic training completed")
    
    def recognize_word(self, trajectory):
        """Recognize word from trajectory"""
        if not self.is_trained:
            return "unknown", 0.0
            
        processor = TrajectoryProcessor()
        features = processor.extract_features(trajectory).reshape(1, -1)
        features_scaled = self.scaler.transform(features)
        
        # Get prediction and confidence
        prediction = self.classifier.predict(features_scaled)[0]
        probabilities = self.classifier.predict_proba(features_scaled)[0]
        confidence = np.max(probabilities)
        
        # Apply auto-correction using TextBlob
        corrected = str(TextBlob(prediction).correct())
        
        return corrected, confidence
    
    def save_model(self):
        """Save trained model"""
        model_data = {
            'classifier': self.classifier,
            'scaler': self.scaler,
            'words': self.words,
            'is_trained': self.is_trained
        }
        with open('airwriting_model.pkl', 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self):
        """Load pre-trained model"""
        try:
            with open('airwriting_model.pkl', 'rb') as f:
                model_data = pickle.load(f)
                self.classifier = model_data['classifier']
                self.scaler = model_data['scaler']
                self.words = model_data['words']
                self.is_trained = model_data['is_trained']
                print("✅ Loaded pre-trained model")
        except FileNotFoundError:
            print("⚠️ No pre-trained model found")

class VoiceFeedback:
    """Handles text-to-speech feedback"""
    def __init__(self):
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)
        self.engine.setProperty('volume', 0.8)
        
        # Set voice to female if available
        voices = self.engine.getProperty('voices')
        for voice in voices:
            if 'female' in voice.name.lower() or 'zira' in voice.name.lower():
                self.engine.setProperty('voice', voice.id)
                break
    
    def speak(self, text):
        """Speak text in a separate thread"""
        def _speak():
            self.engine.say(text)
            self.engine.runAndWait()
        
        thread = threading.Thread(target=_speak)
        thread.daemon = True
        thread.start()

class AirWritingSystem:
    """Main Air Writing Recognition System"""
    def __init__(self):
        # Initialize components
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Initialize modules
        self.filter = MovingAverageFilter(window_size=5)
        self.gesture_detector = GestureDetector(threshold=0.05)
        self.word_recognizer = WordRecognizer()
        self.voice_feedback = VoiceFeedback()
        
        # State variables
        self.trajectory = []
        self.canvas = None
        self.is_tracking = False
        self.last_recognition_time = 0
        self.recognition_cooldown = 2.0  # seconds
        self.status = "Ready"
        self.recognized_word = ""
        self.confidence = 0.0
        
        # UI colors
        self.colors = {
            'trajectory': (0, 255, 0),      # Green
            'text': (0, 0, 255),            # Red
            'status': (255, 255, 0),        # Yellow
            'landmarks': (255, 0, 255)      # Magenta
        }
        
        print("🚀 Air Writing System initialized!")
        print("📋 Instructions:")
        print("   • Open hand = Start tracking")
        print("   • Close hand (fist) = Stop & recognize")
        print("   • Press 'C' = Clear canvas")
        print("   • Press 'ESC' = Exit")
        print("   • Try words: cat, dog, sun, run, big, red, yes, no, top, box")
    
    def process_frame(self, frame):
        """Process a single frame"""
        height, width = frame.shape[:2]
        
        # Initialize canvas if needed
        if self.canvas is None:
            self.canvas = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks
                self.mp_draw.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                )
                
                # Get index finger tip position
                index_tip = hand_landmarks.landmark[8]
                x = int(index_tip.x * width)
                y = int(index_tip.y * height)
                
                # Apply smoothing filter
                x_smooth, y_smooth = self.filter.filter(x, y)
                x_smooth, y_smooth = int(x_smooth), int(y_smooth)
                
                # Check hand gesture
                is_open = self.gesture_detector.is_hand_open(hand_landmarks.landmark)
                
                if is_open:
                    # Hand is open - track movement
                    if not self.is_tracking:
                        self.is_tracking = True
                        self.trajectory = []
                        self.status = "✋ Tracking"
                    
                    # Add point to trajectory
                    self.trajectory.append([x_smooth, y_smooth])
                    
                    # Draw on canvas
                    if len(self.trajectory) > 1:
                        cv2.line(self.canvas, 
                                tuple(self.trajectory[-2]), 
                                tuple(self.trajectory[-1]), 
                                self.colors['trajectory'], 3)
                
                else:
                    # Hand is closed - stop tracking and recognize
                    if self.is_tracking and len(self.trajectory) > 10:
                        self.is_tracking = False
                        self.status = "🤚 Recognizing..."
                        
                        # Recognize word
                        current_time = time.time()
                        if current_time - self.last_recognition_time > self.recognition_cooldown:
                            self.recognize_trajectory()
                            self.last_recognition_time = current_time
                    
                    elif not self.is_tracking:
                        self.status = "🤚 Paused"
        
        else:
            # No hand detected
            self.is_tracking = False
            self.status = "👋 Show your hand"
        
        # Combine frame and canvas
        combined = cv2.addWeighted(frame, 0.7, self.canvas, 0.3, 0)
        
        # Add UI elements
        self.draw_ui(combined)
        
        return combined
    
    def recognize_trajectory(self):
        """Recognize word from current trajectory"""
        if len(self.trajectory) < 5:
            return
        
        word, confidence = self.word_recognizer.recognize_word(self.trajectory)
        
        if confidence > 0.3:  # Minimum confidence threshold
            self.recognized_word = word
            self.confidence = confidence
            
            # Speak the word
            self.voice_feedback.speak(word)
            
            print(f"✅ Recognized: '{word}' (confidence: {confidence:.2f})")
        else:
            self.recognized_word = "unclear"
            self.confidence = confidence
            print(f"❓ Unclear trajectory (confidence: {confidence:.2f})")
        
        # Clear canvas after recognition
        self.clear_canvas()
    
    def draw_ui(self, frame):
        """Draw user interface elements"""
        height, width = frame.shape[:2]
        
        # Status
        cv2.putText(frame, f"Status: {self.status}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['status'], 2)
        
        # Recognized word
        if self.recognized_word:
            text = f"Word: {self.recognized_word} ({self.confidence:.2f})"
            cv2.putText(frame, text, 
                       (10, height - 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, self.colors['text'], 2)
        
        # Instructions
        cv2.putText(frame, "Open hand = Track | Close hand = Recognize | C = Clear | ESC = Exit", 
                   (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Trajectory count
        if self.is_tracking:
            cv2.putText(frame, f"Points: {len(self.trajectory)}", 
                       (width - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['trajectory'], 2)
    
    def clear_canvas(self):
        """Clear the drawing canvas"""
        if self.canvas is not None:
            self.canvas.fill(0)
        self.trajectory = []
        self.recognized_word = ""
        self.confidence = 0.0
        print("🧹 Canvas cleared")
    
    def run(self):
        """Main application loop"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Error: Could not open camera")
            return
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("🎥 Camera started. Begin air writing!")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("❌ Error: Could not read frame")
                    break
                
                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Process frame
                processed_frame = self.process_frame(frame)
                
                # Display frame
                cv2.imshow('Air Writing System', processed_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC key
                    break
                elif key == ord('c') or key == ord('C'):
                    self.clear_canvas()
        
        except KeyboardInterrupt:
            print("\n👋 Interrupted by user")
        
        finally:
            # Cleanup
            cap.release()
            cv2.destroyAllWindows()
            print("🏁 Air Writing System finished!")

def main():
    """Main function"""
    print("🖐️ REAL-TIME AIR WRITING RECOGNITION SYSTEM")
    print("=" * 50)
    
    try:
        system = AirWritingSystem()
        system.run()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()