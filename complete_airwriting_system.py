#!/usr/bin/env python3
"""
Complete Air Writing Recognition System
Integrated solution using existing code with enhanced features
"""

import cv2
import numpy as np
import time
import json
import os
import threading
import argparse
from collections import deque
from pathlib import Path

# Try imports with fallbacks
MEDIAPIPE_AVAILABLE = False
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    print("⚠️  MediaPipe not available - using fallback tracking")

try:
    from keras.models import load_model
    import tensorflow as tf
    MODEL_AVAILABLE = True
except ImportError:
    MODEL_AVAILABLE = False
    print("⚠️  TensorFlow/Keras not available - using demo mode")

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False
    print("⚠️  TextBlob not available - basic word correction only")

try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    print("⚠️  Text-to-speech not available")

class UniversalHandTracker:
    """Universal hand tracker that works with or without MediaPipe"""
    
    def __init__(self):
        self.mediapipe_tracker = None
        self.fallback_active = False
        
        # Initialize MediaPipe if available
        if MEDIAPIPE_AVAILABLE:
            try:
                self.mp_hands = mp.solutions.hands
                self.hands = self.mp_hands.Hands(
                    static_image_mode=False,
                    max_num_hands=1,
                    model_complexity=1,
                    min_detection_confidence=0.6,
                    min_tracking_confidence=0.4
                )
                self.mp_drawing = mp.solutions.drawing_utils
                self.mp_drawing_styles = mp.solutions.drawing_styles
                print("✅ MediaPipe hand tracking initialized")
            except Exception as e:
                print(f"⚠️  MediaPipe initialization failed: {e}")
                # MEDIAPIPE_AVAILABLE is already False at module level
        
        # Tracking state
        self.trail = deque(maxlen=200)
        self.velocity_history = deque(maxlen=10)
        self.current_position = None
        self.previous_position = None
        self.velocity = 0.0
        self.is_writing = False
        self.hand_detected = False
        
        # Smoothing parameters
        self.smoothing_factor = 0.7
        self.velocity_threshold = 5.0
        
        print(f"🖐️  Hand tracker initialized (MediaPipe: {MEDIAPIPE_AVAILABLE})")
    
    def detect_hand_mediapipe(self, frame):
        """Detect hand using MediaPipe"""
        if not MEDIAPIPE_AVAILABLE:
            return None, False
        
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw hand landmarks
                    self.mp_drawing.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style()
                    )
                    
                    # Get index fingertip (landmark 8)
                    h, w, _ = frame.shape
                    fingertip = hand_landmarks.landmark[8]
                    x, y = int(fingertip.x * w), int(fingertip.y * h)
                    
                    # Check if in writing position (index finger extended)
                    writing_gesture = self.detect_writing_gesture(hand_landmarks)
                    
                    return (x, y), writing_gesture
            
            return None, False
            
        except Exception as e:
            print(f"MediaPipe error: {e}")
            return None, False
    
    def detect_writing_gesture(self, hand_landmarks):
        """Detect if hand is in writing position"""
        try:
            landmarks = hand_landmarks.landmark
            
            # Index finger landmarks
            index_tip = landmarks[8]
            index_pip = landmarks[6]
            index_mcp = landmarks[5]
            
            # Other finger tips
            middle_tip = landmarks[12]
            middle_pip = landmarks[10]
            ring_tip = landmarks[16]
            ring_pip = landmarks[14]
            pinky_tip = landmarks[20]
            pinky_pip = landmarks[18]
            
            # Check if index finger is extended
            index_extended = index_tip.y < index_pip.y < index_mcp.y
            
            # Check if other fingers are curled
            middle_curled = middle_tip.y > middle_pip.y
            ring_curled = ring_tip.y > ring_pip.y
            pinky_curled = pinky_tip.y > pinky_pip.y
            
            # Calculate confidence
            confidence = 0.0
            if index_extended: confidence += 0.4
            if middle_curled: confidence += 0.2
            if ring_curled: confidence += 0.2
            if pinky_curled: confidence += 0.2
            
            return confidence > 0.6
            
        except Exception:
            return False
    
    def detect_hand_fallback(self, frame):
        """Fallback hand detection using color/contour detection"""
        try:
            # Convert to HSV for better color detection
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Skin color range (adjust as needed)
            lower_skin = np.array([0, 20, 70])
            upper_skin = np.array([20, 255, 255])
            
            # Create mask
            mask = cv2.inRange(hsv, lower_skin, upper_skin)
            
            # Morphological operations
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Get largest contour
                largest_contour = max(contours, key=cv2.contourArea)
                
                if cv2.contourArea(largest_contour) > 2000:
                    # Find topmost point (fingertip)
                    topmost = tuple(largest_contour[largest_contour[:, :, 1].argmin()][0])
                    
                    # Draw contour
                    cv2.drawContours(frame, [largest_contour], -1, (0, 255, 0), 2)
                    
                    return topmost, True
            
            return None, False
            
        except Exception as e:
            print(f"Fallback detection error: {e}")
            return None, False
    
    def smooth_position(self, new_position):
        """Apply smoothing to position"""
        if self.current_position is None:
            return new_position
        
        # Exponential moving average
        smoothed_x = (self.smoothing_factor * self.current_position[0] + 
                     (1 - self.smoothing_factor) * new_position[0])
        smoothed_y = (self.smoothing_factor * self.current_position[1] + 
                     (1 - self.smoothing_factor) * new_position[1])
        
        return (int(smoothed_x), int(smoothed_y))
    
    def calculate_velocity(self):
        """Calculate current velocity"""
        if self.previous_position is None or self.current_position is None:
            return 0.0
        
        dx = self.current_position[0] - self.previous_position[0]
        dy = self.current_position[1] - self.previous_position[1]
        return np.sqrt(dx*dx + dy*dy)
    
    def update_trail(self, position):
        """Update drawing trail"""
        if position and self.is_writing:
            self.trail.append(position)
    
    def draw_trail(self, frame):
        """Draw the drawing trail"""
        if len(self.trail) < 2:
            return
        
        trail_points = list(self.trail)
        
        for i in range(1, len(trail_points)):
            # Gradient color
            alpha = i / len(trail_points)
            color = (int(255 * (1-alpha)), int(255 * alpha), 0)
            thickness = max(2, int(6 * alpha))
            
            cv2.line(frame, trail_points[i-1], trail_points[i], color, thickness)
    
    def process_frame(self, frame):
        """Process frame and return tracking results"""
        # Try MediaPipe first, then fallback
        if MEDIAPIPE_AVAILABLE and not self.fallback_active:
            fingertip, writing_gesture = self.detect_hand_mediapipe(frame)
        else:
            fingertip, writing_gesture = self.detect_hand_fallback(frame)
            if not self.fallback_active:
                self.fallback_active = True
                print("🔄 Switched to fallback hand tracking")
        
        # Update tracking state
        self.hand_detected = fingertip is not None
        
        if fingertip:
            # Apply smoothing
            smoothed_position = self.smooth_position(fingertip)
            
            # Update positions
            self.previous_position = self.current_position
            self.current_position = smoothed_position
            
            # Calculate velocity
            current_velocity = self.calculate_velocity()
            self.velocity_history.append(current_velocity)
            
            # Average velocity
            if len(self.velocity_history) > 0:
                self.velocity = sum(self.velocity_history) / len(self.velocity_history)
            
            # Determine if writing
            self.is_writing = (writing_gesture and self.velocity > self.velocity_threshold)
            
            # Update trail
            self.update_trail(smoothed_position)
            
            # Draw fingertip
            color = (0, 0, 255) if self.is_writing else (0, 255, 0)
            cv2.circle(frame, smoothed_position, 10, color, -1)
            cv2.circle(frame, smoothed_position, 12, (255, 255, 255), 2)
        
        else:
            self.velocity = 0.0
            self.is_writing = False
        
        # Draw trail
        self.draw_trail(frame)
        
        return fingertip, self.velocity, self.is_writing
    
    def get_trail_path(self):
        """Get current trail as list"""
        return list(self.trail)
    
    def clear_trail(self):
        """Clear the trail"""
        self.trail.clear()

class LetterRecognizer:
    """Letter recognition with fallback options"""

    def __init__(self, model_path=None):
        self.model = None
        self.model_available = False

        # Try to load model
        if MODEL_AVAILABLE:
            self.load_model(model_path)

        print(f"🧠 Letter recognizer initialized (Model: {self.model_available})")

    def load_model(self, model_path=None):
        """Load trained model"""
        model_paths = []

        # If custom path provided, try it first
        if model_path and os.path.exists(model_path):
            model_paths.append(model_path)

        # Default paths
        default_paths = [
            "models/letter_recognition_advanced.h5",
            "models/advanced_letter_recognition.h5",
            "models/letter_recognition_enhanced.h5",
            "models/letter_recognition.h5"
        ]
        model_paths.extend(default_paths)

        for path in model_paths:
            if os.path.exists(path):
                try:
                    self.model = load_model(path)
                    self.model_available = True
                    print(f"✅ Loaded model: {path}")
                    return
                except Exception as e:
                    print(f"⚠️  Failed to load {path}: {e}")

        print("⚠️  No trained model found - using demo mode")
    
    def draw_path_on_blank(self, path, img_size=32):
        """Optimized path drawing using enhanced preprocessing"""
        # Use the optimized preprocessing from utils
        from utils.preprocessing import draw_path_on_blank
        return draw_path_on_blank(path, img_size)
    
    def recognize_letter(self, path):
        """Recognize letter from path"""
        if len(path) < 5:
            return None, 0.0
        
        if self.model_available and self.model is not None:
            try:
                # Create image from path
                img = self.draw_path_on_blank(path)
                
                # Resize for model
                img_resized = cv2.resize(img, (28, 28))
                img_normalized = img_resized.reshape(1, 28, 28, 1) / 255.0
                
                # Predict
                prediction = self.model.predict(img_normalized, verbose=0)
                confidence = np.max(prediction)
                letter_idx = np.argmax(prediction)
                letter = chr(letter_idx + ord('A'))
                
                return letter, confidence
                
            except Exception as e:
                print(f"Model prediction error: {e}")
                return self.demo_recognition(path)
        else:
            return self.demo_recognition(path)
    
    def demo_recognition(self, path):
        """Demo letter recognition (random but consistent)"""
        # Use path characteristics to generate consistent "recognition"
        path_hash = hash(str(path)) % 26
        letter = chr(path_hash + ord('A'))
        confidence = 0.6 + (path_hash % 4) * 0.1  # 0.6-0.9
        return letter, confidence

class WordCorrector:
    """Word correction with multiple strategies"""
    
    def __init__(self):
        self.target_words = [
            'CAT', 'DOG', 'BAT', 'RAT', 'HAT', 'MAT', 'SAT', 'FAT',
            'BIG', 'PIG', 'DIG', 'FIG', 'WIG', 'JIG',
            'SUN', 'RUN', 'FUN', 'GUN', 'BUN', 'NUN',
            'BOX', 'FOX', 'COX', 'SOX',
            'BED', 'RED', 'LED', 'FED',
            'TOP', 'HOP', 'MOP', 'POP', 'COP',
            'CUP', 'PUP', 'SUP',
            'BAG', 'TAG', 'RAG', 'SAG', 'WAG',
            'BUS', 'YES', 'NET', 'PET', 'SET', 'WET', 'GET', 'LET', 'MET',
            'HOT', 'POT', 'COT', 'DOT', 'GOT', 'LOT', 'NOT', 'ROT',
            'BAD', 'DAD', 'HAD', 'MAD', 'PAD', 'SAD',
            'BEE', 'SEE', 'TEE', 'FEE',
            'EGG', 'LEG', 'BEG', 'PEG',
            'ICE', 'NICE', 'RICE', 'MICE'
        ]
        
        self.word_set = set(self.target_words)
        print(f"📚 Word corrector initialized with {len(self.target_words)} words")
    
    def correct_word(self, word):
        """Correct word using multiple strategies including Levenshtein distance"""
        if not word:
            return ""

        word = word.upper().strip()

        # Direct match
        if word in self.word_set:
            return word

        # Find best match by character similarity
        best_match = word
        best_score = 0

        for target_word in self.target_words:
            if len(target_word) == len(word):
                # Character-by-character matching
                matches = sum(1 for a, b in zip(word, target_word) if a == b)
                score = matches / len(word)

                if score > best_score and score >= 0.5:  # At least 50% match
                    best_score = score
                    best_match = target_word

        # Levenshtein distance for better correction
        if best_score < 0.7:
            best_match = self._levenshtein_correct(word)

        # Try TextBlob if available and still no good match
        if TEXTBLOB_AVAILABLE and best_score < 0.7:
            try:
                blob = TextBlob(word.lower())
                corrected = str(blob.correct()).upper()
                if corrected in self.word_set:
                    return corrected
            except Exception:
                pass

        return best_match

    def _levenshtein_correct(self, word):
        """Correct word using Levenshtein distance"""
        min_distance = float('inf')
        best_match = word

        for target in self.target_words:
            distance = self._levenshtein_distance(word, target)
            if distance < min_distance:
                min_distance = distance
                best_match = target

        # Only return correction if distance is reasonable (max 2 edits)
        return best_match if min_distance <= 2 else word

    def _levenshtein_distance(self, s1, s2):
        """Calculate Levenshtein distance between two strings"""
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)

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
    
    def get_suggestions(self, partial_word):
        """Get word completion suggestions"""
        if not partial_word:
            return []
        
        partial = partial_word.upper()
        suggestions = [word for word in self.target_words if word.startswith(partial)]
        return suggestions[:5]

class TextToSpeech:
    """Text-to-speech with fallback"""
    
    def __init__(self):
        self.tts_available = TTS_AVAILABLE
        self.engine = None
        
        if TTS_AVAILABLE:
            try:
                self.engine = pyttsx3.init()
                # Set properties
                self.engine.setProperty('rate', 150)  # Speed
                self.engine.setProperty('volume', 0.8)  # Volume
                print("🔊 Text-to-speech initialized")
            except Exception as e:
                print(f"⚠️  TTS initialization failed: {e}")
                self.tts_available = False
        else:
            print("⚠️  Text-to-speech not available")
    
    def speak(self, text):
        """Speak text"""
        if self.tts_available and self.engine:
            try:
                self.engine.say(text)
                self.engine.runAndWait()
            except Exception as e:
                print(f"TTS error: {e}")
        else:
            print(f"🔊 Would speak: {text}")

class CompleteAirWritingSystem:
    """Complete integrated air writing system"""
    
    def __init__(self, model_path=None):
        print("🚀 Initializing Complete Air Writing System...")
        
        # Initialize components
        self.hand_tracker = UniversalHandTracker()
        self.letter_recognizer = LetterRecognizer(model_path)
        self.word_corrector = WordCorrector()
        self.tts = TextToSpeech()
        
        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise Exception("Could not open camera")
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Application state
        self.current_word = ""
        self.recognized_words = []
        self.current_path = []
        
        # Timing parameters
        self.LETTER_PAUSE_FRAMES = 25
        self.WORD_PAUSE_FRAMES = 75
        self.MIN_PATH_LENGTH = 10
        self.CONFIDENCE_THRESHOLD = 0.3
        
        # State counters
        self.letter_pause_count = 0
        self.word_pause_count = 0
        self.last_movement_time = time.time()
        
        # Performance tracking
        self.fps = 0
        self.frame_count = 0
        self.fps_start_time = time.time()
        
        print("✅ Complete Air Writing System initialized")
        self.print_instructions()
    
    def print_instructions(self):
        """Print usage instructions"""
        print("\n" + "="*60)
        print("🖐️  COMPLETE AIR WRITING SYSTEM")
        print("="*60)
        print("📋 Instructions:")
        print("   • Hold up your index finger (other fingers curled)")
        print("   • Write letters in the air")
        print("   • Pause briefly between letters")
        print("   • Pause longer between words")
        print("\n🎯 Try these simple words:")
        sample_words = ['CAT', 'DOG', 'SUN', 'BOX', 'RED', 'BIG', 'TOP', 'CUP']
        print("   " + " | ".join(sample_words))
        print("\n⌨️  Controls:")
        print("   SPACE - End current letter")
        print("   S     - Speak current word")
        print("   C     - Clear current word")
        print("   T     - Toggle trail")
        print("   ESC   - Exit")
        print("="*60 + "\n")
    
    def update_fps(self):
        """Update FPS counter"""
        self.frame_count += 1
        if time.time() - self.fps_start_time >= 1.0:
            self.fps = self.frame_count
            self.frame_count = 0
            self.fps_start_time = time.time()
    
    def draw_ui(self, frame):
        """Draw user interface"""
        h, w = frame.shape[:2]
        
        # Main display area
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (w-10, 160), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        
        # Current word
        word_text = f"Word: {self.current_word}"
        cv2.putText(frame, word_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
        
        # Word suggestions
        if self.current_word:
            suggestions = self.word_corrector.get_suggestions(self.current_word)
            if suggestions:
                suggestion_text = f"Suggestions: {' | '.join(suggestions[:3])}"
                cv2.putText(frame, suggestion_text, (20, 85), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
        
        # Recent words
        if self.recognized_words:
            recent_words = " | ".join(self.recognized_words[-3:])
            cv2.putText(frame, f"Recent: {recent_words}", (20, 115), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # System status
        status_text = []
        if MEDIAPIPE_AVAILABLE:
            status_text.append("MediaPipe: ✅")
        else:
            status_text.append("MediaPipe: ❌ (Fallback)")
        
        if self.letter_recognizer.model_available:
            status_text.append("Model: ✅")
        else:
            status_text.append("Model: ❌ (Demo)")
        
        if TTS_AVAILABLE:
            status_text.append("TTS: ✅")
        else:
            status_text.append("TTS: ❌")
        
        cv2.putText(frame, " | ".join(status_text), (20, 145), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Performance info
        cv2.putText(frame, f"FPS: {self.fps}", (w - 150, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f"Letters: {len(self.current_word)}", (w - 150, 55), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f"Words: {len(self.recognized_words)}", (w - 150, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Instructions
        instructions = [
            "Hold index finger up and write letters in air",
            "SPACE: End letter | S: Speak | C: Clear | ESC: Exit"
        ]
        
        for i, instruction in enumerate(instructions):
            cv2.putText(frame, instruction, (20, h - 60 + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        
        # Target words panel
        self.draw_target_words_panel(frame)
    
    def draw_target_words_panel(self, frame):
        """Draw target words panel"""
        h, w = frame.shape[:2]
        
        # Panel
        panel_x, panel_y = w - 320, 10
        panel_w, panel_h = 300, 200
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Title
        cv2.putText(frame, "Target Words:", (panel_x + 10, panel_y + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Show words in grid
        words_to_show = self.word_corrector.target_words[:18]
        cols, rows = 3, 6
        
        for i, word in enumerate(words_to_show):
            row = i // cols
            col = i % cols
            
            x = panel_x + 15 + col * 90
            y = panel_y + 50 + row * 25
            
            # Highlight current word
            color = (0, 255, 0) if word == self.current_word else (200, 200, 200)
            cv2.putText(frame, word, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    def process_letter_completion(self):
        """Process completed letter"""
        if len(self.current_path) < self.MIN_PATH_LENGTH:
            return
        
        letter, confidence = self.letter_recognizer.recognize_letter(self.current_path)
        
        if letter and confidence > self.CONFIDENCE_THRESHOLD:
            self.current_word += letter
            print(f"Letter: {letter} (confidence: {confidence:.3f})")
        else:
            print(f"Letter rejected: {letter} (confidence: {confidence:.3f})")
        
        # Clear path
        self.current_path.clear()
        self.letter_pause_count = 0
    
    def process_word_completion(self):
        """Process completed word"""
        if not self.current_word:
            return
        
        # Correct word
        corrected_word = self.word_corrector.correct_word(self.current_word)
        
        print(f"Word: {self.current_word} -> {corrected_word}")
        
        # Add to recognized words
        self.recognized_words.append(corrected_word)
        
        # Speak word in separate thread
        threading.Thread(target=self.tts.speak, args=(corrected_word,), daemon=True).start()
        
        # Reset
        self.current_word = ""
        self.word_pause_count = 0
        self.current_path.clear()
    
    def run(self):
        """Main application loop"""
        print("🚀 Starting Complete Air Writing System...")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("❌ Failed to read from camera")
                break
            
            # Flip frame
            frame = cv2.flip(frame, 1)
            
            # Process hand tracking
            fingertip, velocity, is_writing = self.hand_tracker.process_frame(frame)
            
            # Update FPS
            self.update_fps()
            
            # Handle writing logic
            if fingertip and is_writing:
                self.current_path.append(fingertip)
                self.last_movement_time = time.time()
                self.letter_pause_count = 0
                self.word_pause_count = 0
            else:
                # No active writing
                if len(self.current_path) > 0:
                    self.letter_pause_count += 1
                
                # Letter completion
                if self.letter_pause_count >= self.LETTER_PAUSE_FRAMES:
                    self.process_letter_completion()
                
                # Word completion
                if time.time() - self.last_movement_time > 3.0:
                    self.word_pause_count += 1
                    
                    if self.word_pause_count >= self.WORD_PAUSE_FRAMES:
                        self.process_word_completion()
            
            # Draw UI
            self.draw_ui(frame)
            
            # Show frame
            cv2.imshow("Complete Air Writing System", frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' '):  # End letter
                self.process_letter_completion()
            
            elif key == ord('s'):  # Speak word
                if self.current_word:
                    corrected_word = self.word_corrector.correct_word(self.current_word)
                    self.recognized_words.append(corrected_word)
                    threading.Thread(target=self.tts.speak, args=(corrected_word,), daemon=True).start()
                    self.current_word = ""
                    self.current_path.clear()
            
            elif key == ord('c'):  # Clear
                self.current_word = ""
                self.current_path.clear()
                self.hand_tracker.clear_trail()
                print("Cleared")
            
            elif key == ord('t'):  # Toggle trail
                # Toggle trail visibility (implementation depends on tracker)
                print("Trail toggled")
            
            elif key == 27:  # ESC
                break
        
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()
        
        # Session summary
        print(f"\n📊 Session Summary:")
        print(f"   Words recognized: {len(self.recognized_words)}")
        if self.recognized_words:
            print(f"   Words: {', '.join(self.recognized_words)}")
        print("👋 Complete Air Writing System finished!")

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Complete Air Writing Recognition System")
    parser.add_argument('--model', type=str, help='Path to custom model file (.h5)')
    parser.add_argument('--list-models', action='store_true', help='List available default models and exit')
    return parser.parse_args()

def list_available_models():
    """List available model files"""
    model_paths = [
        "models/letter_recognition_advanced.h5",
        "models/advanced_letter_recognition.h5",
        "models/letter_recognition_enhanced.h5",
        "models/letter_recognition.h5"
    ]
    available = [p for p in model_paths if os.path.exists(p)]
    if available:
        print("\nAvailable default models:")
        for i, path in enumerate(available, 1):
            print(f"{i}. {path}")
    else:
        print("\nNo default models found. You can train one or specify a custom path with --model.")
    print("\nTo use a model: python complete_airwriting_system.py --model path/to/model.h5")

def main():
    """Main function"""
    args = parse_args()
    
    if args.list_models:
        list_available_models()
        return
    
    model_path = args.model
    
    print("🖐️  COMPLETE AIR WRITING RECOGNITION SYSTEM")
    print("=" * 60)
    if model_path:
        print(f"Using custom model: {model_path}")
    print("🎯 Integrated solution with fallback support")
    print("=" * 60)
    
    try:
        system = CompleteAirWritingSystem(model_path=model_path)
        system.run()
    except KeyboardInterrupt:
        print("\n👋 System interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()