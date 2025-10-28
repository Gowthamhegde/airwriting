#!/usr/bin/env python3
"""
Complete Air Writing Recognition System
Advanced MediaPipe-based real-time air writing with word recognition and voice feedback
"""

import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque
import math
import threading
import os
import json
from datetime import datetime

# Text processing and auto-correction
try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False
    print("⚠️  TextBlob not available - basic correction only")

# Text-to-speech
try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    print("⚠️  Text-to-speech not available")

# Machine learning for letter recognition
try:
    from tensorflow.keras.models import load_model
    import tensorflow as tf
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("⚠️  TensorFlow not available - using demo recognition")

class AdvancedHandTracker:
    """Advanced MediaPipe-based hand tracker with smooth trails and gesture recognition"""
    
    def __init__(self):
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Hand detection settings
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.8,
            min_tracking_confidence=0.7,
            model_complexity=1
        )
        
        # Movement tracking with enhanced smoothing
        self.trail_points = deque(maxlen=200)  # Longer trail for better letters
        self.fingertip_trail = deque(maxlen=50)
        self.velocity_history = deque(maxlen=15)
        
        # Hand landmarks
        self.current_landmarks = None
        self.fingertip_position = None
        self.palm_center = None
        
        # Enhanced movement analysis
        self.is_writing = False
        self.movement_threshold = 8
        self.writing_velocity_threshold = 12
        self.smoothed_velocity = 0.0
        
        # Gesture detection
        self.finger_states = {'thumb': False, 'index': False, 'middle': False, 'ring': False, 'pinky': False}
        self.gesture_name = "Unknown"
        self.gesture_confidence = 0.0
        
        # Smoothing parameters
        self.position_alpha = 0.3
        self.velocity_alpha = 0.4
        self.prev_position = None
        
    def detect_finger_states(self, landmarks):
        """Detect which fingers are extended"""
        if not landmarks:
            return
        
        # Finger tip and pip landmarks
        finger_tips = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky
        finger_pips = [3, 6, 10, 14, 18]
        
        fingers = []
        
        # Thumb (different logic due to orientation)
        if landmarks[finger_tips[0]].x > landmarks[finger_pips[0]].x:
            fingers.append(1)
        else:
            fingers.append(0)
        
        # Other fingers
        for i in range(1, 5):
            if landmarks[finger_tips[i]].y < landmarks[finger_pips[i]].y:
                fingers.append(1)
            else:
                fingers.append(0)
        
        # Update finger states
        finger_names = ['thumb', 'index', 'middle', 'ring', 'pinky']
        for i, name in enumerate(finger_names):
            self.finger_states[name] = bool(fingers[i])
        
        # Determine gesture
        self.determine_gesture(fingers)
        
        return fingers
    
    def determine_gesture(self, fingers):
        """Determine hand gesture based on finger states"""
        total_fingers = sum(fingers)
        
        if total_fingers == 0:
            self.gesture_name = "Fist"
        elif total_fingers == 1:
            if fingers[1]:  # Index finger only
                self.gesture_name = "Pointing"
            elif fingers[0]:  # Thumb only
                self.gesture_name = "Thumbs Up"
            else:
                self.gesture_name = "One Finger"
        elif total_fingers == 2:
            if fingers[1] and fingers[2]:  # Index and middle
                self.gesture_name = "Peace/Victory"
            elif fingers[0] and fingers[1]:  # Thumb and index
                self.gesture_name = "Gun/L-Shape"
            else:
                self.gesture_name = "Two Fingers"
        elif total_fingers == 5:
            self.gesture_name = "Open Hand"
        else:
            self.gesture_name = f"{total_fingers} Fingers"
    
    def get_fingertip_position(self, landmarks, frame_shape):
        """Get index fingertip position in pixel coordinates"""
        if landmarks:
            # Index finger tip (landmark 8)
            index_tip = landmarks[8]
            h, w = frame_shape[:2]
            x = int(index_tip.x * w)
            y = int(index_tip.y * h)
            return (x, y)
        return None
    
    def get_palm_center(self, landmarks, frame_shape):
        """Get palm center position"""
        if landmarks:
            # Use wrist (0) and middle finger MCP (9) to estimate palm center
            wrist = landmarks[0]
            middle_mcp = landmarks[9]
            
            h, w = frame_shape[:2]
            center_x = int((wrist.x + middle_mcp.x) / 2 * w)
            center_y = int((wrist.y + middle_mcp.y) / 2 * h)
            return (center_x, center_y)
        return None
    
    def calculate_hand_velocity(self):
        """Calculate hand movement velocity"""
        if len(self.fingertip_trail) < 2:
            return 0
        
        # Calculate distance between last two positions
        p1 = self.fingertip_trail[-2]
        p2 = self.fingertip_trail[-1]
        
        distance = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
        self.velocity_history.append(distance)
        
        # Calculate average velocity
        avg_velocity = np.mean(self.velocity_history) if self.velocity_history else 0
        
        # Determine if writing motion
        self.is_writing = avg_velocity > self.writing_velocity_threshold
        
        return avg_velocity
    
    def process_frame(self, frame):
        """Process frame and detect hand landmarks"""
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = self.hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            # Get first hand
            hand_landmarks = results.multi_hand_landmarks[0]
            self.current_landmarks = hand_landmarks.landmark
            
            # Get fingertip and palm positions
            self.fingertip_position = self.get_fingertip_position(self.current_landmarks, frame.shape)
            self.palm_center = self.get_palm_center(self.current_landmarks, frame.shape)
            
            # Detect finger states and gestures
            self.detect_finger_states(self.current_landmarks)
            
            # Determine if writing (index finger extended, others curled)
            writing_gesture = (self.finger_states['index'] and 
                             not self.finger_states['middle'] and 
                             not self.finger_states['ring'] and 
                             not self.finger_states['pinky'])
            
            # Calculate velocity
            velocity = self.calculate_hand_velocity()
            
            # Determine if actively writing (gesture + movement)
            self.is_writing = writing_gesture and velocity > self.writing_velocity_threshold
            
            # Update trails
            if self.fingertip_position and self.is_writing:
                # Add to fingertip trail
                if (not self.fingertip_trail or 
                    math.sqrt((self.fingertip_position[0] - self.fingertip_trail[-1][0])**2 + 
                             (self.fingertip_position[1] - self.fingertip_trail[-1][1])**2) > self.movement_threshold):
                    self.fingertip_trail.append(self.fingertip_position)
                
                # Add to main trail for letter recognition
                self.trail_points.append(self.fingertip_position)
            
            # Return fingertip position, velocity, writing status, and gesture
            return self.fingertip_position, velocity, self.is_writing, self.gesture_name
        else:
            self.current_landmarks = None
            self.fingertip_position = None
            self.palm_center = None
            self.is_writing = False
            return None, 0, False, "none"
    
    def draw_hand_landmarks(self, frame, show_connections=True):
        """Draw hand landmarks and connections"""
        if not self.current_landmarks:
            return
        
        # Create a hand landmarks object for MediaPipe drawing
        from mediapipe.framework.formats import landmark_pb2
        
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        for landmark in self.current_landmarks:
            hand_landmarks_proto.landmark.append(landmark)
        
        # Draw landmarks and connections
        if show_connections:
            self.mp_drawing.draw_landmarks(
                frame, 
                hand_landmarks_proto,
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                self.mp_drawing_styles.get_default_hand_connections_style()
            )
        else:
            self.mp_drawing.draw_landmarks(
                frame, 
                hand_landmarks_proto,
                None,
                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                None
            )

class HandTrackingSystem:
    """Main MediaPipe hand tracking system"""
    
    def __init__(self):
        self.tracker = MediaPipeHandTracker()
        self.cap = None
        self.running = False
        
        # Display settings
        self.show_landmarks = True
        self.show_trail = True
        self.show_fingertip_trail = True
        self.show_connections = True
        self.show_info = True
        
        # Colors
        self.trail_color = (0, 255, 0)  # Green
        self.fingertip_trail_color = (255, 0, 0)  # Red
        self.fingertip_color = (0, 0, 255)  # Red
        self.palm_color = (255, 255, 0)  # Cyan
        
    def initialize_camera(self):
        """Initialize camera"""
        print("🎥 Initializing camera...")
        self.cap = cv2.VideoCapture(0)
        
        if not self.cap.isOpened():
            print("❌ Could not open camera")
            return False
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("✅ Camera initialized")
        return True
    
    def draw_trails(self, frame):
        """Draw movement trails"""
        # Draw main trail (writing trail)
        if self.show_trail and len(self.tracker.trail_points) > 1:
            for i in range(1, len(self.tracker.trail_points)):
                thickness = max(1, int((i / len(self.tracker.trail_points)) * 5))
                cv2.line(frame, 
                        self.tracker.trail_points[i-1], 
                        self.tracker.trail_points[i], 
                        self.trail_color, 
                        thickness)
        
        # Draw fingertip trail
        if self.show_fingertip_trail and len(self.tracker.fingertip_trail) > 1:
            for i in range(1, len(self.tracker.fingertip_trail)):
                alpha = i / len(self.tracker.fingertip_trail)
                thickness = max(1, int(alpha * 3))
                cv2.line(frame, 
                        self.tracker.fingertip_trail[i-1], 
                        self.tracker.fingertip_trail[i], 
                        self.fingertip_trail_color, 
                        thickness)
    
    def draw_hand_info(self, frame, velocity):
        """Draw hand information overlay"""
        if not self.show_info:
            return
        
        height, width = frame.shape[:2]
        
        # Hand detection status
        if self.tracker.current_landmarks:
            status_text = "✅ Hand Detected"
            status_color = (0, 255, 0)
        else:
            status_text = "❌ No Hand"
            status_color = (0, 0, 255)
        
        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        if self.tracker.current_landmarks:
            # Gesture information
            gesture_text = f"Gesture: {self.tracker.gesture_name}"
            cv2.putText(frame, gesture_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Finger states
            finger_text = "Fingers: "
            for name, state in self.tracker.finger_states.items():
                finger_text += f"{name[0].upper()}{'✓' if state else '✗'} "
            cv2.putText(frame, finger_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Position information
            if self.tracker.fingertip_position:
                pos_text = f"Fingertip: {self.tracker.fingertip_position}"
                cv2.putText(frame, pos_text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Velocity and writing status
            vel_text = f"Velocity: {velocity:.1f}"
            cv2.putText(frame, vel_text, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            writing_text = "✍️ Writing Motion" if self.tracker.is_writing else "✋ Stationary"
            writing_color = (0, 255, 255) if self.tracker.is_writing else (255, 255, 255)
            cv2.putText(frame, writing_text, (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, writing_color, 1)
        
        # Controls
        controls = [
            "Controls:",
            "L - Toggle landmarks",
            "T - Toggle trail", 
            "F - Toggle fingertip trail",
            "C - Toggle connections",
            "I - Toggle info",
            "R - Reset trails",
            "ESC - Exit"
        ]
        
        for i, control in enumerate(controls):
            cv2.putText(frame, control, (width - 250, 30 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    def draw_landmarks_and_connections(self, frame, hand_landmarks):
        """Draw hand landmarks and connections"""
        if not hand_landmarks:
            return
        
        # Draw landmarks
        if self.show_landmarks:
            self.tracker.mp_drawing.draw_landmarks(
                frame, 
                hand_landmarks,
                self.tracker.mp_hands.HAND_CONNECTIONS if self.show_connections else None,
                self.tracker.mp_drawing_styles.get_default_hand_landmarks_style(),
                self.tracker.mp_drawing_styles.get_default_hand_connections_style() if self.show_connections else None
            )
        
        # Draw special points
        if self.tracker.fingertip_position:
            cv2.circle(frame, self.tracker.fingertip_position, 8, self.fingertip_color, -1)
            cv2.circle(frame, self.tracker.fingertip_position, 12, (255, 255, 255), 2)
        
        if self.tracker.palm_center:
            cv2.circle(frame, self.tracker.palm_center, 6, self.palm_color, -1)
    
    def run(self):
        """Run the MediaPipe hand tracking system"""
        if not self.initialize_camera():
            return
        
        print("\n🖐️  MEDIAPIPE HAND TRACKING SYSTEM")
        print("=" * 60)
        print("📋 Instructions:")
        print("   • Hold your hand in front of the camera")
        print("   • Point with index finger to draw trail")
        print("   • Try different gestures (fist, peace, open hand)")
        print("   • Red trail shows fingertip movement")
        print("   • Green trail shows writing motion")
        print("\n⌨️  Controls:")
        print("   L - Toggle landmarks display")
        print("   T - Toggle writing trail")
        print("   F - Toggle fingertip trail")
        print("   C - Toggle hand connections")
        print("   I - Toggle info overlay")
        print("   R - Reset all trails")
        print("   ESC - Exit")
        print("=" * 60)
        
        self.running = True
        fps_counter = 0
        fps_start_time = time.time()
        
        try:
            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    print("❌ Failed to read frame")
                    break
                
                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Process frame with MediaPipe
                hand_landmarks, velocity = self.tracker.process_frame(frame)
                
                # Draw trails
                self.draw_trails(frame)
                
                # Draw hand landmarks and connections
                self.draw_landmarks_and_connections(frame, hand_landmarks)
                
                # Draw information overlay
                self.draw_hand_info(frame, velocity)
                
                # Show frame
                cv2.imshow('MediaPipe Hand Tracking', frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == 27:  # ESC
                    break
                elif key == ord('l') or key == ord('L'):
                    self.show_landmarks = not self.show_landmarks
                    print(f"Landmarks display: {'ON' if self.show_landmarks else 'OFF'}")
                elif key == ord('t') or key == ord('T'):
                    self.show_trail = not self.show_trail
                    print(f"Writing trail: {'ON' if self.show_trail else 'OFF'}")
                elif key == ord('f') or key == ord('F'):
                    self.show_fingertip_trail = not self.show_fingertip_trail
                    print(f"Fingertip trail: {'ON' if self.show_fingertip_trail else 'OFF'}")
                elif key == ord('c') or key == ord('C'):
                    self.show_connections = not self.show_connections
                    print(f"Hand connections: {'ON' if self.show_connections else 'OFF'}")
                elif key == ord('i') or key == ord('I'):
                    self.show_info = not self.show_info
                    print(f"Info overlay: {'ON' if self.show_info else 'OFF'}")
                elif key == ord('r') or key == ord('R'):
                    self.tracker.trail_points.clear()
                    self.tracker.fingertip_trail.clear()
                    print("All trails reset")
                
                # Calculate and display FPS
                fps_counter += 1
                if fps_counter % 30 == 0:
                    fps = 30 / (time.time() - fps_start_time)
                    print(f"📊 FPS: {fps:.1f} | Gesture: {self.tracker.gesture_name}")
                    fps_start_time = time.time()
        
        except KeyboardInterrupt:
            print("\n👋 System interrupted by user")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        self.running = False
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("🧹 Cleanup completed")

class LetterRecognizer:
    """Enhanced letter recognition with ML model support"""
    
    def __init__(self):
        self.model = None
        self.model_available = False
        
        # Try to load trained model
        if ML_AVAILABLE:
            self.load_model()
        
        print(f"🧠 Letter recognizer initialized (Model: {self.model_available})")
    
    def load_model(self):
        """Load trained letter recognition model"""
        model_paths = [
            "models/letter_recognition.h5",
            "models/letter_recognition_enhanced.h5",
            "models/letter_recognition_advanced.h5"
        ]
        
        for path in model_paths:
            if os.path.exists(path):
                try:
                    self.model = load_model(path, compile=False)
                    self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
                    self.model_available = True
                    print(f"✅ Loaded model: {path}")
                    return
                except Exception as e:
                    print(f"⚠️  Failed to load {path}: {e}")
        
        print("⚠️  No trained model found - using demo recognition")
    
    def path_to_image(self, path, img_size=28):
        """Convert path points to image for recognition"""
        if len(path) < 3:
            return None
        
        # Create blank image
        img = np.ones((img_size, img_size), dtype=np.uint8) * 255
        
        # Normalize path to image size
        path_array = np.array(path)
        min_x, min_y = np.min(path_array, axis=0)
        max_x, max_y = np.max(path_array, axis=0)
        
        width = max_x - min_x
        height = max_y - min_y
        
        if width <= 0 or height <= 0:
            return None
        
        # Scale to fit in image with padding
        padding = 4
        scale = min((img_size - 2 * padding) / width, (img_size - 2 * padding) / height)
        
        # Center the path
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        offset_x = img_size / 2 - center_x * scale
        offset_y = img_size / 2 - center_y * scale
        
        # Draw path
        scaled_path = []
        for x, y in path:
            new_x = int(x * scale + offset_x)
            new_y = int(y * scale + offset_y)
            new_x = np.clip(new_x, 0, img_size - 1)
            new_y = np.clip(new_y, 0, img_size - 1)
            scaled_path.append((new_x, new_y))
        
        # Draw lines between points
        for i in range(1, len(scaled_path)):
            cv2.line(img, scaled_path[i-1], scaled_path[i], (0), 2)
        
        # Light blur for smoothing
        img = cv2.GaussianBlur(img, (3, 3), 0.5)
        
        return img
    
    def recognize_letter(self, path):
        """Recognize letter from path"""
        if len(path) < 5:
            return None, 0.0
        
        if self.model_available and self.model is not None:
            try:
                # Convert path to image
                img = self.path_to_image(path)
                if img is None:
                    return None, 0.0
                
                # Prepare for model
                img_normalized = img.reshape(1, 28, 28, 1) / 255.0
                
                # Predict
                prediction = self.model.predict(img_normalized, verbose=0)
                confidence = np.max(prediction)
                letter_idx = np.argmax(prediction)
                letter = chr(letter_idx + ord('A'))
                
                return letter, confidence
                
            except Exception as e:
                print(f"Recognition error: {e}")
                return self.demo_recognition(path)
        else:
            return self.demo_recognition(path)
    
    def demo_recognition(self, path):
        """Demo letter recognition for testing"""
        # Simple heuristic based on path characteristics
        path_array = np.array(path)
        
        # Calculate basic features
        width = np.max(path_array[:, 0]) - np.min(path_array[:, 0])
        height = np.max(path_array[:, 1]) - np.min(path_array[:, 1])
        aspect_ratio = width / height if height > 0 else 1.0
        
        # Simple letter mapping based on aspect ratio and path length
        path_length = len(path)
        
        if aspect_ratio > 1.5:  # Wide letters
            letters = ['A', 'H', 'M', 'N', 'W']
        elif aspect_ratio < 0.7:  # Tall letters
            letters = ['I', 'L', 'T', 'F', 'E']
        else:  # Square-ish letters
            letters = ['O', 'C', 'D', 'P', 'B', 'R', 'S']
        
        # Use path characteristics to pick letter
        letter_idx = (path_length + int(width) + int(height)) % len(letters)
        letter = letters[letter_idx]
        confidence = 0.6 + (letter_idx % 4) * 0.1  # 0.6-0.9
        
        return letter, confidence

class WordCorrector:
    """Advanced word correction with multiple strategies"""
    
    def __init__(self):
        # Common 3-letter words for air writing
        self.target_words = [
            'CAT', 'DOG', 'BAT', 'RAT', 'HAT', 'MAT', 'SAT', 'FAT', 'PAT',
            'BIG', 'PIG', 'DIG', 'FIG', 'WIG', 'JIG', 'RIG',
            'SUN', 'RUN', 'FUN', 'GUN', 'BUN', 'NUN',
            'BOX', 'FOX', 'COX', 'SOX',
            'BED', 'RED', 'LED', 'FED', 'WED',
            'TOP', 'HOP', 'MOP', 'POP', 'COP', 'SOP',
            'CUP', 'PUP', 'SUP',
            'BAG', 'TAG', 'RAG', 'SAG', 'WAG', 'NAG',
            'BUS', 'YES', 'NEW', 'OLD', 'HOT', 'COT', 'DOT', 'GOT', 'LOT', 'NOT', 'ROT',
            'BAD', 'DAD', 'HAD', 'MAD', 'PAD', 'SAD',
            'BEE', 'SEE', 'TEE', 'FEE',
            'EGG', 'LEG', 'BEG', 'PEG',
            'ICE', 'ACE', 'AGE',
            'ANT', 'PAN', 'MAN', 'CAN', 'FAN', 'VAN',
            'CAR', 'FAR', 'JAR', 'BAR', 'TAR',
            'DAY', 'BAY', 'HAY', 'JAY', 'LAY', 'MAY', 'PAY', 'RAY', 'SAY', 'WAY',
            'EYE', 'DYE', 'RYE', 'SKY', 'FLY', 'TRY'
        ]
        
        self.word_set = set(self.target_words)
        
        # Create frequency map (higher frequency = more common)
        self.word_frequencies = {}
        for i, word in enumerate(self.target_words):
            # 3-letter words get higher frequency
            if len(word) == 3:
                freq = 1.0 - (i * 0.01)  # Decreasing frequency
            elif len(word) == 4:
                freq = 0.8 - (i * 0.005)
            else:
                freq = 0.6 - (i * 0.002)
            self.word_frequencies[word] = max(0.1, freq)
        
        # Group words by length for faster matching
        self.words_by_length = {}
        for word in self.target_words:
            length = len(word)
            if length not in self.words_by_length:
                self.words_by_length[length] = []
            self.words_by_length[length].append(word)
        
        print(f"📚 Loaded dictionary with {len(self.target_words)} words")
    
    def levenshtein_distance(self, s1, s2):
        """Calculate Levenshtein distance between two strings"""
        if len(s1) < len(s2):
            return self.levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    def correct_word(self, word):
        """Correct word using multiple strategies"""
        if not word:
            return ""
        
        word = word.upper().strip()
        
        # Direct match
        if word in self.word_set:
            return word
        
        # Find best matches using Levenshtein distance
        candidates = []
        word_length = len(word)
        
        # Check words of similar length first
        for length in range(max(1, word_length - 1), word_length + 2):
            if length in self.words_by_length:
                for target_word in self.words_by_length[length]:
                    distance = self.levenshtein_distance(word, target_word)
                    # Calculate similarity score
                    max_len = max(len(word), len(target_word))
                    similarity = 1.0 - (distance / max_len) if max_len > 0 else 0.0
                    candidates.append((target_word, similarity))
        
        # If no similar length words, check all
        if not candidates:
            for target_word in self.target_words:
                distance = self.levenshtein_distance(word, target_word)
                max_len = max(len(word), len(target_word))
                similarity = 1.0 - (distance / max_len) if max_len > 0 else 0.0
                candidates.append((target_word, similarity))
        
        # Sort by similarity and return best match
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        if candidates and candidates[0][1] > 0.3:  # Minimum similarity threshold
            return candidates[0][0]
        
        # Try TextBlob if available
        if TEXTBLOB_AVAILABLE:
            try:
                blob = TextBlob(word.lower())
                corrected = str(blob.correct()).upper()
                if corrected in self.word_set:
                    return corrected
            except:
                pass
        
        return word  # Return original if no good correction found
    
    def get_suggestions(self, partial_word, max_suggestions=3):
        """Get word completion suggestions"""
        if not partial_word:
            return []
        
        partial = partial_word.upper()
        suggestions = []
        
        # Find words that start with the partial word
        for word in self.target_words:
            if word.startswith(partial):
                suggestions.append(word)
        
        return suggestions[:max_suggestions]

class VoiceFeedback:
    """Enhanced text-to-speech voice feedback system with smart management"""
    
    def __init__(self):
        self.tts_available = TTS_AVAILABLE
        self.engine = None
        self.speaking = False
        self.recent_words = deque(maxlen=5)
        self.last_speech_time = 0
        self.min_speech_interval = 1.0  # Minimum seconds between speeches
        
        # Try to initialize multiple TTS engines
        self.engines_available = []
        self.current_engine = None
        
        # Try pyttsx3 first
        if TTS_AVAILABLE:
            try:
                self.engine = pyttsx3.init()
                self.configure_pyttsx3()
                self.engines_available.append('pyttsx3')
                self.current_engine = 'pyttsx3'
                print("🔊 pyttsx3 text-to-speech initialized")
            except Exception as e:
                print(f"⚠️  pyttsx3 TTS initialization failed: {e}")
                self.tts_available = False
        
        # Try gTTS as fallback
        try:
            from gtts import gTTS
            import pygame
            pygame.mixer.init()
            self.engines_available.append('gTTS')
            if not self.current_engine:
                self.current_engine = 'gTTS'
            print("🔊 gTTS + pygame available as fallback")
        except ImportError:
            pass
        
        if not self.engines_available:
            print("⚠️  No text-to-speech engines available")
            self.tts_available = False
    
    def configure_pyttsx3(self):
        """Configure pyttsx3 engine settings"""
        if not self.engine:
            return
        
        try:
            # Set speech rate (words per minute)
            self.engine.setProperty('rate', 180)  # Slightly faster for better UX
            
            # Set volume (0.0 to 1.0)
            self.engine.setProperty('volume', 0.9)
            
            # Try to set a clear voice
            voices = self.engine.getProperty('voices')
            if voices:
                # Prefer female voice if available
                for voice in voices:
                    if voice.name and ('female' in voice.name.lower() or 
                                     'zira' in voice.name.lower() or
                                     'hazel' in voice.name.lower()):
                        self.engine.setProperty('voice', voice.id)
                        print(f"✅ Voice set to: {voice.name}")
                        break
                else:
                    # Use first available voice
                    self.engine.setProperty('voice', voices[0].id)
                    print(f"✅ Voice set to: {voices[0].name}")
        except Exception as e:
            print(f"⚠️  pyttsx3 configuration error: {e}")
    
    def should_speak(self, word):
        """Check if word should be spoken (avoid repetition)"""
        current_time = time.time()
        
        # Check minimum interval
        if current_time - self.last_speech_time < self.min_speech_interval:
            return False
        
        # Check if word was recently spoken
        if word in self.recent_words:
            return False
        
        return True
    
    def speak_word_pyttsx3(self, word):
        """Speak using pyttsx3 engine"""
        if not self.engine:
            return
        
        try:
            self.engine.say(word)
            self.engine.runAndWait()
        except Exception as e:
            print(f"pyttsx3 speech error: {e}")
    
    def speak_word_gtts(self, word):
        """Speak using gTTS + pygame"""
        try:
            from gtts import gTTS
            import pygame
            import io
            
            # Generate speech with gTTS
            tts = gTTS(text=word, lang='en', slow=False)
            
            # Save to memory buffer
            audio_buffer = io.BytesIO()
            tts.write_to_fp(audio_buffer)
            audio_buffer.seek(0)
            
            # Play with pygame
            pygame.mixer.music.load(audio_buffer)
            pygame.mixer.music.play()
            
            # Wait for playback to finish
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
                
        except Exception as e:
            print(f"gTTS speech error: {e}")
    
    def speak_word(self, word):
        """Speak a word with smart management (non-blocking)"""
        if not self.tts_available or not word or self.speaking:
            return
        
        word = word.strip().upper()
        
        # Check if we should speak this word
        if not self.should_speak(word):
            return
        
        def speak_thread():
            try:
                self.speaking = True
                
                if self.current_engine == 'pyttsx3':
                    self.speak_word_pyttsx3(word)
                elif self.current_engine == 'gTTS':
                    self.speak_word_gtts(word)
                else:
                    print(f"🔊 Would speak: {word}")
                
                # Update history
                self.recent_words.append(word)
                self.last_speech_time = time.time()
                
            except Exception as e:
                print(f"Speech error: {e}")
            finally:
                self.speaking = False
        
        # Run in separate thread to avoid blocking
        threading.Thread(target=speak_thread, daemon=True).start()
    
    def is_speaking(self):
        """Check if currently speaking"""
        return self.speaking
    
    def get_engine_info(self):
        """Get information about available engines"""
        return {
            'available_engines': self.engines_available,
            'current_engine': self.current_engine,
            'is_speaking': self.speaking,
            'recent_words': list(self.recent_words)
        }

class CompleteAirWritingSystem:
    """Complete air writing recognition system with all advanced features"""
    
    def __init__(self):
        print("🚀 Initializing Complete Air Writing System...")
        
        # Initialize all components
        self.hand_tracker = AdvancedHandTracker()
        self.letter_recognizer = LetterRecognizer()
        self.word_corrector = WordCorrector()
        self.voice_feedback = VoiceFeedback()
        
        # Initialize camera with optimized settings for 30+ FPS
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise Exception("Could not open camera")
        
        # Optimized camera settings for real-time performance
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce latency
        
        # Application state
        self.current_word = ""
        self.recognized_words = []
        self.current_path = []
        self.word_suggestions = []
        
        # Enhanced timing parameters for smooth experience
        self.letter_pause_frames = 20   # Reduced for better responsiveness
        self.word_pause_frames = 60     # Frames to wait before completing word
        self.min_path_length = 8        # Minimum path length for letter recognition
        self.idle_time_threshold = 1.5  # Seconds of idle time to complete word
        
        # State counters
        self.letter_pause_count = 0
        self.word_pause_count = 0
        self.last_movement_time = time.time()
        self.last_word_spoken_time = 0
        
        # Enhanced display settings
        self.show_trail = True
        self.show_landmarks = True
        self.show_suggestions = True
        self.trail_fade_effect = True
        self.glow_effect = True
        self.animated_trails = True
        self.background_blur = False
        self.show_debug = False
        
        # Animation state for word completion effects
        self.word_animation_active = False
        self.word_animation_start = 0
        self.word_animation_duration = 1.0
        self.animation_word = ""
        
        # Performance tracking for 30+ FPS
        self.fps = 0
        self.frame_count = 0
        self.fps_start_time = time.time()
        self.target_fps = 30
        self.processing_times = deque(maxlen=30)
        
        # Color schemes for trails
        self.color_schemes = {
            'gradient': [(255, 0, 0), (0, 255, 255)],  # Blue to Red
            'rainbow': [(255, 0, 0), (255, 127, 0), (255, 255, 0), (0, 255, 0), (0, 0, 255)],
            'fire': [(0, 0, 255), (0, 127, 255), (0, 255, 255), (255, 255, 255)],
            'ocean': [(255, 255, 0), (255, 127, 0), (255, 0, 0)]
        }
        self.current_color_scheme = 'gradient'
        
        # Output logging
        self.output_log = []
        self.output_file = "output_log.txt"
        
        print("✅ Complete Air Writing System initialized with enhanced features")
        self.print_instructions()
    
    def print_instructions(self):
        """Print comprehensive usage instructions"""
        print("\n" + "="*80)
        print("🖐️  COMPLETE AIR WRITING RECOGNITION SYSTEM - ENHANCED")
        print("="*80)
        print("📋 Instructions:")
        print("   • Hold up your INDEX FINGER (other fingers curled)")
        print("   • Write letters in the air slowly and clearly")
        print("   • Pause briefly between letters (system auto-detects)")
        print("   • Pause longer between words (1-2 seconds)")
        print("   • System will auto-correct and speak recognized words")
        print("   • Use OPEN HAND gesture to clear the canvas")
        print("\n🎯 Try these words:")
        sample_words = ['CAT', 'DOG', 'SUN', 'BOX', 'RED', 'BIG', 'TOP', 'CUP', 'FISH', 'BIRD']
        print("   " + " | ".join(sample_words))
        print("\n⌨️  Controls:")
        print("   SPACE    - Force complete current letter")
        print("   ENTER    - Force complete current word")
        print("   C        - Clear current word")
        print("   R        - Reset everything")
        print("   T        - Toggle trail display")
        print("   L        - Toggle hand landmarks")
        print("   S        - Toggle word suggestions")
        print("   D        - Toggle debug information")
        print("   B        - Toggle background blur")
        print("   G        - Toggle glow effects")
        print("   A        - Toggle trail animation")
        print("   1-4      - Change trail color schemes")
        print("   ESC      - Exit system")
        print("\n🎨 Visual Effects:")
        print("   1 - Gradient (Blue to Red)    3 - Fire (Red to White)")
        print("   2 - Rainbow colors            4 - Ocean (Cyan to Blue)")
        print("\n🖐️ Gesture Controls:")
        print("   Index finger extended - Writing mode")
        print("   Open hand (all fingers) - Clear canvas")
        print("   Fist - Pause/stop writing")
        print("="*80 + "\n")
    
    def interpolate_color(self, color1, color2, factor):
        """Interpolate between two colors"""
        factor = max(0, min(1, factor))
        return tuple(int(c1 + (c2 - c1) * factor) for c1, c2 in zip(color1, color2))
    
    def get_trail_color(self, progress, scheme='gradient'):
        """Get color for trail based on progress (0-1)"""
        colors = self.color_schemes.get(scheme, self.color_schemes['gradient'])
        
        if len(colors) == 2:
            return self.interpolate_color(colors[0], colors[1], progress)
        
        # Multi-color gradient
        segment_size = 1.0 / (len(colors) - 1)
        segment_index = int(progress / segment_size)
        segment_index = min(segment_index, len(colors) - 2)
        
        local_progress = (progress - segment_index * segment_size) / segment_size
        return self.interpolate_color(colors[segment_index], colors[segment_index + 1], local_progress)
    
    def draw_glowing_trail(self, frame):
        """Draw smooth, glowing, animated trail with enhanced effects"""
        if not self.show_trail or len(self.hand_tracker.trail_points) < 2:
            return
        
        trail_points = list(self.hand_tracker.trail_points)
        
        # Create overlay for glow effect
        overlay = frame.copy()
        
        # Draw multiple layers for glow effect
        glow_layers = 3 if self.glow_effect else 1
        
        for layer in range(glow_layers):
            layer_alpha = 0.3 + layer * 0.2
            base_thickness = 8 - layer * 2
            
            for i in range(1, len(trail_points)):
                # Calculate progress along trail
                progress = i / len(trail_points)
                
                # Get color based on scheme and progress
                color = self.get_trail_color(progress, self.current_color_scheme)
                
                # Apply fade effect
                if self.trail_fade_effect:
                    fade_factor = progress * layer_alpha
                    color = tuple(int(c * fade_factor) for c in color)
                
                # Calculate thickness with animation
                thickness = base_thickness
                if self.animated_trails:
                    # Add subtle thickness animation
                    animation_factor = 1 + 0.3 * math.sin(time.time() * 2 + i * 0.1)
                    thickness = max(1, int(thickness * animation_factor))
                
                # Draw line segment
                cv2.line(overlay, trail_points[i-1], trail_points[i], color, max(1, thickness))
        
        # Blend overlay with original frame
        if self.glow_effect:
            cv2.addWeighted(frame, 0.7, overlay, 0.3, 0, frame)
        else:
            frame[:] = overlay
        
        # Draw animated fingertip with enhanced effects
        if self.hand_tracker.fingertip_position:
            # Pulsing effect
            pulse_factor = abs(math.sin(time.time() * 5))
            pulse_radius = int(8 + pulse_factor * 6)
            
            # Color based on writing state
            base_color = (0, 0, 255) if self.hand_tracker.is_writing else (0, 255, 0)
            
            # Outer glow
            glow_color = tuple(int(c * (0.5 + pulse_factor * 0.5)) for c in base_color)
            cv2.circle(frame, self.hand_tracker.fingertip_position, pulse_radius + 4, glow_color, 2)
            
            # Main circle
            cv2.circle(frame, self.hand_tracker.fingertip_position, pulse_radius, base_color, -1)
            
            # Inner highlight
            cv2.circle(frame, self.hand_tracker.fingertip_position, max(2, pulse_radius - 3), (255, 255, 255), 2)
            
            # Status indicator
            if self.hand_tracker.is_writing:
                cv2.putText(frame, "✍", (self.hand_tracker.fingertip_position[0] + 15, 
                           self.hand_tracker.fingertip_position[1] - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, base_color, 2)
    
    def draw_word_recognition_ui(self, frame):
        """Draw enhanced UI with word recognition info and performance metrics"""
        h, w = frame.shape[:2]
        
        # Create semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (w-10, 220), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        
        # Current word with enhanced animation
        word_display = self.current_word if self.current_word else "[Writing...]"
        if self.current_word:
            word_color = (0, 255, 255)  # Cyan
            # Add pulsing effect for active writing
            if self.hand_tracker.is_writing:
                pulse = abs(math.sin(time.time() * 3)) * 0.3 + 0.7
                word_color = tuple(int(c * pulse) for c in word_color)
        else:
            word_color = (128, 128, 128)  # Gray
        
        cv2.putText(frame, f"Current Word: {word_display}", (20, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.3, word_color, 3)
        
        # Word suggestions with dynamic display
        if self.show_suggestions and self.word_suggestions:
            suggestions_text = f"Suggestions: {' | '.join(self.word_suggestions)}"
            cv2.putText(frame, suggestions_text, (20, 85), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Recent words history
        if self.recognized_words:
            recent_words = " → ".join(self.recognized_words[-4:])
            cv2.putText(frame, f"Recent: {recent_words}", (20, 115), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # System status indicators
        status_y = 145
        
        # Hand detection status
        hand_status = "✅ Hand Detected" if self.hand_tracker.current_landmarks else "❌ No Hand"
        hand_color = (0, 255, 0) if self.hand_tracker.current_landmarks else (0, 0, 255)
        cv2.putText(frame, hand_status, (20, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, hand_color, 2)
        
        # Writing status
        writing_status = "✍️ Writing" if self.hand_tracker.is_writing else "✋ Ready"
        writing_color = (0, 255, 255) if self.hand_tracker.is_writing else (255, 255, 255)
        cv2.putText(frame, writing_status, (200, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, writing_color, 2)
        
        # Voice status
        voice_status = "🔊 Speaking" if self.voice_feedback.is_speaking() else "🔇 Ready"
        voice_color = (255, 0, 255) if self.voice_feedback.is_speaking() else (255, 255, 255)
        cv2.putText(frame, voice_status, (350, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, voice_color, 2)
        
        # Performance metrics
        perf_y = 175
        cv2.putText(frame, f"FPS: {self.fps}", (20, perf_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Processing time
        if self.processing_times:
            avg_proc_time = sum(self.processing_times) / len(self.processing_times)
            cv2.putText(frame, f"Processing: {avg_proc_time:.1f}ms", (120, perf_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.putText(frame, f"Path Points: {len(self.current_path)}", (280, perf_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.putText(frame, f"Words: {len(self.recognized_words)}", (420, perf_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Gesture indicator
        if self.hand_tracker.gesture_name != "Unknown" and self.hand_tracker.gesture_name != "none":
            gesture_y = 205
            gesture_color = (0, 255, 255) if self.hand_tracker.gesture_name == "Open Hand" else (255, 255, 255)
            cv2.putText(frame, f"Gesture: {self.hand_tracker.gesture_name}", (20, gesture_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, gesture_color, 2)
            
            # Special indicator for clear gesture
            if self.hand_tracker.gesture_name == "Open Hand":
                cv2.putText(frame, "🖐 CLEAR", (200, gesture_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        # Debug information
        if self.show_debug:
            debug_y = h - 80
            velocity = self.hand_tracker.smoothed_velocity
            cv2.putText(frame, f"Debug - Velocity: {velocity:.1f} | Scheme: {self.current_color_scheme}", 
                       (20, debug_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Enhanced instructions
        instructions = [
            "✋ Hold INDEX finger up and write letters clearly in the air",
            "⌨️ SPACE: Complete letter | ENTER: Complete word | C: Clear | Open hand: Clear | ESC: Exit"
        ]
        
        for i, instruction in enumerate(instructions):
            cv2.putText(frame, instruction, (20, h - 50 + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    
    def word_completion_animation(self, frame, word, animation_progress=0.0):
        """Show enhanced animation when word is completed"""
        if not word or animation_progress <= 0:
            return
        
        h, w = frame.shape[:2]
        
        # Flash effect intensity (peaks at 0.3, fades out)
        if animation_progress < 0.3:
            flash_intensity = animation_progress / 0.3
        else:
            flash_intensity = 1.0 - (animation_progress - 0.3) / 0.7
        
        flash_intensity = max(0, min(1, flash_intensity))
        
        # Create flash overlay
        if flash_intensity > 0:
            overlay = frame.copy()
            flash_color = (0, 255, 0)  # Green flash
            cv2.rectangle(overlay, (0, 0), (w, h), flash_color, -1)
            cv2.addWeighted(frame, 1 - flash_intensity * 0.3, overlay, flash_intensity * 0.3, 0, frame)
        
        # Animated text display
        if animation_progress > 0.1:  # Start text after initial flash
            text_progress = min(1.0, (animation_progress - 0.1) / 0.6)
            
            # Calculate text size and position with growing effect
            font_scale = 2.0 + text_progress * 1.0  # Growing text
            text_size = cv2.getTextSize(word, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 4)[0]
            text_x = (w - text_size[0]) // 2
            text_y = (h + text_size[1]) // 2
            
            # Text color with fade-in
            text_alpha = text_progress
            text_color = (int(255 * text_alpha), int(255 * text_alpha), int(255 * text_alpha))
            
            # Draw text with outline and glow effect
            cv2.putText(frame, word, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 
                       font_scale, (0, 0, 0), 6)  # Black outline
            cv2.putText(frame, word, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 
                       font_scale, text_color, 4)  # White text
            
            # Add sparkle effect around text
            if text_progress > 0.5:
                sparkle_count = int(10 * (text_progress - 0.5) * 2)
                for _ in range(sparkle_count):
                    sparkle_x = text_x + np.random.randint(-50, text_size[0] + 50)
                    sparkle_y = text_y + np.random.randint(-30, 30)
                    sparkle_size = np.random.randint(2, 6)
                    cv2.circle(frame, (sparkle_x, sparkle_y), sparkle_size, (255, 255, 255), -1)
    
    def process_letter_completion(self):
        """Process completed letter with enhanced recognition and auto-completion"""
        if len(self.current_path) < self.min_path_length:
            return
        
        processing_start = time.time()
        
        # Recognize letter
        letter, confidence = self.letter_recognizer.recognize_letter(self.current_path)
        
        # Track processing time
        processing_time = (time.time() - processing_start) * 1000
        self.processing_times.append(processing_time)
        
        # Adaptive confidence threshold based on path quality
        min_confidence = 0.25 if len(self.current_path) > 20 else 0.3
        
        if letter and confidence > min_confidence:
            self.current_word += letter
            print(f"✅ Letter: {letter} (confidence: {confidence:.3f})")
            
            # Update word suggestions with contextual awareness
            self.word_suggestions = self.word_corrector.get_suggestions(self.current_word, max_suggestions=3)
            
            # Auto-complete check for high-confidence single matches
            if len(self.current_word) >= 3:
                exact_matches = [word for word in self.word_corrector.target_words 
                               if word.startswith(self.current_word)]
                if len(exact_matches) == 1 and confidence > 0.7:
                    completed_word = exact_matches[0]
                    if len(completed_word) == len(self.current_word):
                        # Word is complete, trigger word completion
                        print(f"🤖 Auto-completing: {self.current_word} → {completed_word}")
                        self.process_word_completion()
                        return
        else:
            print(f"❌ Letter rejected: {letter} (confidence: {confidence:.3f})")
        
        # Clear path for next letter
        self.current_path.clear()
        self.hand_tracker.trail_points.clear()
        self.letter_pause_count = 0
    
    def process_word_completion(self):
        """Process completed word with enhanced correction and voice feedback"""
        if not self.current_word:
            return
        
        # Correct the word using enhanced algorithms
        corrected_word = self.word_corrector.correct_word(self.current_word)
        
        print(f"📝 Word: {self.current_word} → {corrected_word}")
        
        # Add to recognized words
        self.recognized_words.append(corrected_word)
        
        # Log to file with timestamp
        self.log_word(corrected_word)
        
        # Trigger word completion animation
        self.word_animation_active = True
        self.word_animation_start = time.time()
        self.animation_word = corrected_word
        
        # Speak the word (non-blocking with smart management)
        current_time = time.time()
        if current_time - self.last_word_spoken_time > 1.0:  # Prevent rapid repetition
            # Use threading to avoid blocking the main loop
            threading.Thread(target=self.voice_feedback.speak_word, args=(corrected_word,), daemon=True).start()
            self.last_word_spoken_time = current_time
        
        # Reset for next word
        self.reset_word_state()
    
    def reset_word_state(self):
        """Reset word-related state"""
        self.current_word = ""
        self.word_suggestions = []
        self.word_pause_count = 0
        self.current_path.clear()
        self.hand_tracker.trail_points.clear()
    
    def log_word(self, word):
        """Log recognized word to file with enhanced formatting"""
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_entry = f"{timestamp}: {word}\n"
            
            with open(self.output_file, "a", encoding="utf-8") as f:
                f.write(log_entry)
            
            self.output_log.append((timestamp, word))
            print(f"📝 Logged: {word}")
        except Exception as e:
            print(f"⚠️ Logging error: {e}")
    
    def apply_background_blur(self, frame):
        """Apply background blur effect to focus on hand writing"""
        if not self.background_blur:
            return frame
        
        # Simple background blur - in practice, you'd use hand segmentation
        blurred = cv2.GaussianBlur(frame, (15, 15), 0)
        return blurred
    
    def update_fps(self):
        """Update FPS counter"""
        self.frame_count += 1
        if time.time() - self.fps_start_time >= 1.0:
            self.fps = self.frame_count
            self.frame_count = 0
            self.fps_start_time = time.time()
    
    def handle_keyboard_input(self, key):
        """Handle keyboard input with comprehensive controls"""
        if key == ord(' '):  # SPACE - Force letter completion
            self.process_letter_completion()
            
        elif key == 13:  # ENTER - Force word completion
            if self.current_word:
                self.process_word_completion()
                
        elif key == ord('c') or key == ord('C'):  # Clear current word
            self.reset_word_state()
            print("🧹 Cleared current word")
            
        elif key == ord('r') or key == ord('R'):  # Reset everything
            self.recognized_words.clear()
            self.reset_word_state()
            print("🔄 Reset everything")
            
        elif key == ord('t') or key == ord('T'):  # Toggle trail
            self.show_trail = not self.show_trail
            print(f"🎨 Trail display: {'ON' if self.show_trail else 'OFF'}")
            
        elif key == ord('l') or key == ord('L'):  # Toggle landmarks
            self.show_landmarks = not self.show_landmarks
            print(f"🖐️ Landmarks: {'ON' if self.show_landmarks else 'OFF'}")
            
        elif key == ord('s') or key == ord('S'):  # Toggle suggestions
            self.show_suggestions = not self.show_suggestions
            print(f"💡 Suggestions: {'ON' if self.show_suggestions else 'OFF'}")
            
        elif key == ord('d') or key == ord('D'):  # Toggle debug
            self.show_debug = not self.show_debug
            print(f"🐛 Debug info: {'ON' if self.show_debug else 'OFF'}")
            
        elif key == ord('b') or key == ord('B'):  # Toggle background blur
            self.background_blur = not self.background_blur
            print(f"🌫️ Background blur: {'ON' if self.background_blur else 'OFF'}")
            
        elif key == ord('g') or key == ord('G'):  # Toggle glow effect
            self.glow_effect = not self.glow_effect
            print(f"✨ Glow effect: {'ON' if self.glow_effect else 'OFF'}")
            
        elif key == ord('a') or key == ord('A'):  # Toggle animation
            self.animated_trails = not self.animated_trails
            print(f"🎬 Trail animation: {'ON' if self.animated_trails else 'OFF'}")
            
        # Color scheme changes
        elif key == ord('1'):
            self.current_color_scheme = 'gradient'
            print("🎨 Color scheme: Gradient (Blue to Red)")
        elif key == ord('2'):
            self.current_color_scheme = 'rainbow'
            print("🎨 Color scheme: Rainbow")
        elif key == ord('3'):
            self.current_color_scheme = 'fire'
            print("🎨 Color scheme: Fire (Red to White)")
        elif key == ord('4'):
            self.current_color_scheme = 'ocean'
            print("🎨 Color scheme: Ocean (Cyan to Blue)")
            
        elif key == 27:  # ESC - Exit
            return False
            
        return True
    
    def run(self):
        """Run the complete air writing system with enhanced performance"""
        print("🚀 Starting Complete Air Writing System...")
        
        # Performance optimization variables
        frame_skip_counter = 0
        process_every_n_frames = 1  # Process every frame for maximum responsiveness
        
        try:
            while True:
                loop_start_time = time.time()
                
                ret, frame = self.cap.read()
                if not ret:
                    print("❌ Failed to read frame")
                    break
                
                # Flip frame for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Apply background blur if enabled
                if self.background_blur:
                    frame = self.apply_background_blur(frame)
                
                # Process hand tracking
                fingertip, velocity, is_writing, gesture = self.hand_tracker.process_frame(frame)
                
                # Update FPS
                self.update_fps()
                
                # Handle writing logic with enhanced timing
                current_time = time.time()
                
                if fingertip and is_writing:
                    # Active writing detected
                    self.current_path.append(fingertip)
                    self.last_movement_time = current_time
                    self.letter_pause_count = 0
                    self.word_pause_count = 0
                    
                else:
                    # No active writing - handle pauses and completions
                    if len(self.current_path) > 0:
                        self.letter_pause_count += 1
                    
                    # Letter completion (pause-based or time-based)
                    time_since_movement = current_time - self.last_movement_time
                    if (self.letter_pause_count >= self.letter_pause_frames or 
                        (len(self.current_path) >= self.min_path_length and time_since_movement > 0.8)):
                        self.process_letter_completion()
                    
                    # Word completion (longer pause or clear gesture)
                    if (time_since_movement > self.idle_time_threshold or 
                        gesture == "Open Hand"):
                        self.word_pause_count += 1
                        
                        if (self.word_pause_count >= self.word_pause_frames or 
                            gesture == "Open Hand" or
                            time_since_movement > 3.0):
                            if self.current_word:
                                self.process_word_completion()
                            elif gesture == "Open Hand":
                                # Clear gesture without word - just clear trail
                                self.hand_tracker.clear_trail()
                                self.current_path.clear()
                                print("🖐️ Clear gesture detected - trail cleared")
                
                # Draw hand landmarks
                if self.show_landmarks:
                    self.hand_tracker.draw_hand_landmarks(frame, show_connections=True)
                
                # Draw glowing trail with effects
                self.draw_glowing_trail(frame)
                
                # Draw UI
                self.draw_word_recognition_ui(frame)
                
                # Draw target words panel
                self.draw_target_words_panel(frame)
                
                # Draw word completion animation
                if self.word_animation_active:
                    animation_progress = (current_time - self.word_animation_start) / self.word_animation_duration
                    if animation_progress <= 1.0:
                        self.word_completion_animation(frame, self.animation_word, animation_progress)
                    else:
                        self.word_animation_active = False
                
                # Display frame
                cv2.imshow('Complete Air Writing System', frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if not self.handle_keyboard_input(key):
                    break
                
                # Performance monitoring and frame rate control
                loop_time = time.time() - loop_start_time
                target_loop_time = 1.0 / self.target_fps
                
                if loop_time < target_loop_time:
                    # Sleep to maintain target FPS
                    time.sleep(target_loop_time - loop_time)
                elif loop_time > target_loop_time * 1.5:
                    # Performance warning
                    if self.fps < 25:
                        print(f"⚠️ Performance warning: FPS = {self.fps}, Loop time = {loop_time*1000:.1f}ms")
        
        except KeyboardInterrupt:
            print("\n👋 System interrupted by user")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources and show comprehensive session summary"""
        # Release camera
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        
        # Stop voice feedback
        if hasattr(self.voice_feedback, 'engine') and self.voice_feedback.engine:
            try:
                self.voice_feedback.engine.stop()
            except:
                pass
        
        # Calculate session duration
        session_duration = time.time() - self.fps_start_time
        
        # Show comprehensive session summary
        print(f"\n📊 SESSION SUMMARY")
        print("=" * 60)
        print(f"   Session duration: {session_duration/60:.1f} minutes")
        print(f"   Words recognized: {len(self.recognized_words)}")
        if self.recognized_words:
            print(f"   Words: {' → '.join(self.recognized_words)}")
        
        # Performance statistics
        if self.processing_times:
            avg_processing = sum(self.processing_times) / len(self.processing_times)
            print(f"   Average processing time: {avg_processing:.1f}ms")
        
        print(f"   Final FPS: {self.fps}")
        print(f"   Color scheme used: {self.current_color_scheme}")
        
        # Feature usage
        features_used = []
        if self.glow_effect: features_used.append("Glow effects")
        if self.animated_trails: features_used.append("Trail animation")
        if self.background_blur: features_used.append("Background blur")
        if features_used:
            print(f"   Visual effects used: {', '.join(features_used)}")
        
        print(f"   Output logged to: {self.output_file}")
        print("=" * 60)
        print("👋 Complete Air Writing System finished!")
        print("   Thank you for using the enhanced air writing system! ✨")
    
    def draw_target_words_panel(self, frame):
        """Draw target words panel with matching highlights"""
        h, w = frame.shape[:2]
        
        # Panel dimensions
        panel_x, panel_y = w - 400, 10
        panel_w, panel_h = 380, 300
        
        # Draw panel background
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        
        # Panel border
        cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (100, 100, 100), 2)
        
        # Title
        cv2.putText(frame, "Target Words", (panel_x + 10, panel_y + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Get words to display (prioritize matching words)
        words_to_show = []
        if self.current_word:
            # Show words that match current input first
            matching_words = [word for word in self.word_corrector.target_words 
                            if word.startswith(self.current_word)]
            words_to_show.extend(matching_words[:15])
            
            # Fill remaining slots with frequent words
            remaining_slots = 24 - len(words_to_show)
            frequent_words = sorted(self.word_corrector.target_words, 
                                  key=lambda w: self.word_corrector.word_frequencies.get(w, 0), 
                                  reverse=True)
            other_words = [word for word in frequent_words[:remaining_slots] 
                          if word not in words_to_show]
            words_to_show.extend(other_words)
        else:
            # Show most frequent words
            frequent_words = sorted(self.word_corrector.target_words, 
                                  key=lambda w: self.word_corrector.word_frequencies.get(w, 0), 
                                  reverse=True)
            words_to_show = frequent_words[:24]
        
        # Display words in grid
        cols, rows = 4, 6
        
        for i, word in enumerate(words_to_show):
            if i >= 24:  # Limit display
                break
                
            row = i // cols
            col = i % cols
            
            x = panel_x + 15 + col * 90
            y = panel_y + 50 + row * 35
            
            # Determine color based on status
            if self.current_word and word.startswith(self.current_word):
                if word == self.current_word:
                    color = (0, 255, 0)  # Exact match - green
                    thickness = 2
                else:
                    color = (0, 255, 255)  # Partial match - cyan
                    thickness = 2
            elif word in self.recognized_words[-5:]:
                color = (255, 0, 255)  # Recently recognized - magenta
                thickness = 2
            else:
                color = (200, 200, 200)  # Default - light gray
                thickness = 1
            
            # Draw word
            font_scale = 0.45 if len(word) <= 4 else 0.35
            cv2.putText(frame, word, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
        
        # Legend
        legend_y = panel_y + panel_h - 50
        cv2.putText(frame, "Legend:", (panel_x + 10, legend_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        legend_items = [
            ("Current", (0, 255, 0)),
            ("Match", (0, 255, 255)),
            ("Recent", (255, 0, 255))
        ]
        
        for i, (label, color) in enumerate(legend_items):
            x_pos = panel_x + 10 + i * 90
            cv2.putText(frame, label, (x_pos, legend_y + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

def main():
    """Main function"""
    print("🚀 Starting Complete Air Writing Recognition System...")
    
    try:
        system = CompleteAirWritingSystem()
        system.run()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("👋 Complete Air Writing System stopped")

if __name__ == "__main__":
    main()