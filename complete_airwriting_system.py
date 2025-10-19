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
        """Load trained model with enhanced error handling"""
        model_paths = []

        # If custom path provided, try it first
        if model_path and os.path.exists(model_path):
            model_paths.append(model_path)

        # Enhanced default paths (prioritize ultra-optimized models)
        default_paths = [
            "models/letter_recognition_ultra_optimized.h5",
            "models/ultra_optimized_letter_recognition.h5",
            "models/letter_recognition_advanced.h5",
            "models/advanced_letter_recognition.h5",
            "models/letter_recognition_enhanced.h5",
            "models/letter_recognition.h5"
        ]
        model_paths.extend(default_paths)

        for path in model_paths:
            if os.path.exists(path):
                try:
                    # Load model with custom objects if needed
                    self.model = load_model(path, compile=False)
                    
                    # Recompile for inference
                    self.model.compile(
                        optimizer='adam',
                        loss='categorical_crossentropy',
                        metrics=['accuracy']
                    )
                    
                    self.model_available = True
                    print(f"✅ Loaded model: {path}")
                    
                    # Test model with a dummy input
                    test_input = np.zeros((1, 28, 28, 1))
                    _ = self.model.predict(test_input, verbose=0)
                    print(f"✅ Model validation successful")
                    return
                    
                except Exception as e:
                    print(f"⚠️  Failed to load {path}: {e}")
                    continue

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
    """Enhanced word correction with ultra-optimized dictionary and advanced algorithms"""
    
    def __init__(self):
        # Load ultra-optimized dictionary if available
        self.dictionary_data = self.load_ultra_dictionary()
        
        if self.dictionary_data:
            self.target_words = self.dictionary_data['target_words']
            self.letter_frequencies = self.dictionary_data['letter_frequencies']
            self.letter_patterns = self.dictionary_data.get('letter_patterns', {})
            print(f"📚 Loaded ultra-optimized dictionary with {len(self.target_words)} words")
        else:
            # Fallback to basic word list
            self.target_words = [
                'CAT', 'DOG', 'BAT', 'RAT', 'HAT', 'MAT', 'SAT', 'FAT', 'PAT',
                'BIG', 'PIG', 'DIG', 'FIG', 'WIG', 'JIG', 'RIG',
                'SUN', 'RUN', 'FUN', 'GUN', 'BUN', 'NUN',
                'BOX', 'FOX', 'COX', 'SOX',
                'BED', 'RED', 'LED', 'FED', 'WED',
                'TOP', 'HOP', 'MOP', 'POP', 'COP', 'SOP',
                'CUP', 'PUP', 'SUP', 'YUP',
                'BAG', 'TAG', 'RAG', 'SAG', 'WAG', 'NAG',
                'BUS', 'HUS', 'MUS',
                'HOT', 'POT', 'COT', 'DOT', 'GOT', 'LOT', 'NOT', 'ROT',
                'BAD', 'DAD', 'HAD', 'MAD', 'PAD', 'SAD', 'FAD',
                'BEE', 'SEE', 'TEE', 'FEE', 'PEE',
                'EGG', 'LEG', 'BEG', 'PEG',
                'ICE', 'NICE', 'RICE', 'MICE', 'VICE', 'DICE',
                'ANT', 'PAN', 'MAN', 'CAN', 'FAN', 'VAN',
                'CAR', 'FAR', 'JAR', 'BAR', 'TAR',
                'DAY', 'BAY', 'HAY', 'JAY', 'LAY', 'MAY', 'PAY', 'RAY', 'SAY', 'WAY',
                'EYE', 'DYE', 'RYE'
            ]
            self.letter_frequencies = {}
            self.letter_patterns = {}
            print(f"📚 Using fallback dictionary with {len(self.target_words)} words")
        
        self.word_set = set(self.target_words)
        
        # Create word length groups for faster matching
        self.words_by_length = {}
        for word in self.target_words:
            length = len(word)
            if length not in self.words_by_length:
                self.words_by_length[length] = []
            self.words_by_length[length].append(word)
    
    def load_ultra_dictionary(self):
        """Load ultra-optimized dictionary if available"""
        try:
            import json
            dict_path = "models/ultra_optimized_word_dictionary.json"
            if os.path.exists(dict_path):
                with open(dict_path, 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"⚠️  Could not load ultra dictionary: {e}")
        return None
    
    def correct_word(self, word):
        """Enhanced word correction using multiple advanced strategies"""
        if not word:
            return ""

        word = word.upper().strip()

        # Direct match - highest priority
        if word in self.word_set:
            return word

        # Length-based optimization - only check words of similar length
        word_length = len(word)
        candidate_words = []
        
        # Check exact length and ±1 length
        for length in range(max(1, word_length - 1), word_length + 2):
            if length in self.words_by_length:
                candidate_words.extend(self.words_by_length[length])
        
        if not candidate_words:
            candidate_words = self.target_words

        # Multi-strategy scoring
        best_match = word
        best_score = 0
        
        for target_word in candidate_words:
            # Calculate multiple similarity scores
            char_score = self._character_similarity(word, target_word)
            position_score = self._positional_similarity(word, target_word)
            levenshtein_score = self._levenshtein_similarity(word, target_word)
            
            # Weighted combination of scores
            combined_score = (
                char_score * 0.4 +
                position_score * 0.3 +
                levenshtein_score * 0.3
            )
            
            # Bonus for exact length match
            if len(target_word) == len(word):
                combined_score *= 1.2
            
            # Bonus for common starting letters
            if word and target_word and word[0] == target_word[0]:
                combined_score *= 1.1
            
            if combined_score > best_score:
                best_score = combined_score
                best_match = target_word

        # Use pattern-based correction if available
        if self.letter_patterns and best_score < 0.6:
            pattern_match = self._pattern_based_correction(word)
            if pattern_match:
                pattern_score = self._character_similarity(word, pattern_match)
                if pattern_score > best_score:
                    best_match = pattern_match

        # Try TextBlob as final fallback
        if TEXTBLOB_AVAILABLE and best_score < 0.5:
            try:
                blob = TextBlob(word.lower())
                corrected = str(blob.correct()).upper()
                if corrected in self.word_set:
                    return corrected
            except Exception:
                pass

        return best_match
    
    def _character_similarity(self, word1, word2):
        """Calculate character-based similarity"""
        if not word1 or not word2:
            return 0.0
        
        matches = sum(1 for a, b in zip(word1, word2) if a == b)
        max_length = max(len(word1), len(word2))
        return matches / max_length if max_length > 0 else 0.0
    
    def _positional_similarity(self, word1, word2):
        """Calculate position-weighted similarity"""
        if not word1 or not word2:
            return 0.0
        
        score = 0.0
        max_length = max(len(word1), len(word2))
        
        for i in range(max_length):
            if i < len(word1) and i < len(word2):
                if word1[i] == word2[i]:
                    # Higher weight for earlier positions
                    weight = 1.0 - (i * 0.1)
                    score += weight
        
        return score / max_length if max_length > 0 else 0.0
    
    def _levenshtein_similarity(self, word1, word2):
        """Calculate Levenshtein-based similarity (0-1 scale)"""
        distance = self._levenshtein_distance(word1, word2)
        max_length = max(len(word1), len(word2))
        return 1.0 - (distance / max_length) if max_length > 0 else 0.0
    
    def _pattern_based_correction(self, word):
        """Use letter patterns for correction"""
        if not self.letter_patterns or len(word) == 0:
            return None
        
        # Find words that match the pattern of the input
        candidates = []
        
        for target_word in self.target_words:
            if len(target_word) == len(word):
                pattern_score = 0
                for i, letter in enumerate(word):
                    if str(i) in self.letter_patterns:
                        pattern_freq = self.letter_patterns[str(i)]
                        if letter in pattern_freq and i < len(target_word):
                            if target_word[i] == letter:
                                pattern_score += pattern_freq[letter]
                
                if pattern_score > 0:
                    candidates.append((target_word, pattern_score))
        
        if candidates:
            # Return the word with highest pattern score
            return max(candidates, key=lambda x: x[1])[0]
        
        return None

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
    """Complete integrated air writing system optimized for real-time performance"""
    
    def __init__(self, model_path=None):
        print("🚀 Initializing Complete Air Writing System...")
        
        # Initialize components
        self.hand_tracker = UniversalHandTracker()
        self.letter_recognizer = LetterRecognizer(model_path)
        self.word_corrector = WordCorrector()
        self.tts = TextToSpeech()
        
        # Initialize camera with optimized settings
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise Exception("Could not open camera")
        
        # Optimized camera properties for real-time performance
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer for real-time
        
        # Application state
        self.current_word = ""
        self.recognized_words = []
        self.current_path = []
        self.letter_candidates = []  # Store multiple letter candidates
        
        # Enhanced timing parameters for better accuracy
        self.LETTER_PAUSE_FRAMES = 20  # Reduced for faster response
        self.WORD_PAUSE_FRAMES = 60   # Reduced for faster word completion
        self.MIN_PATH_LENGTH = 8      # Reduced minimum path length
        self.CONFIDENCE_THRESHOLD = 0.25  # Lowered for better detection
        self.MAX_PATH_LENGTH = 300    # Prevent memory issues
        
        # State counters
        self.letter_pause_count = 0
        self.word_pause_count = 0
        self.last_movement_time = time.time()
        self.last_letter_time = time.time()
        
        # Performance tracking
        self.fps = 0
        self.frame_count = 0
        self.fps_start_time = time.time()
        self.processing_times = deque(maxlen=30)
        
        # Real-time optimization flags
        self.show_trail = True
        self.show_debug = True
        self.auto_word_completion = True
        self.letter_smoothing = True
        
        # Enhanced word prediction
        self.partial_word_predictions = []
        self.word_confidence_history = deque(maxlen=5)
        
        print("✅ Complete Air Writing System initialized with real-time optimizations")
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
        """Enhanced user interface with real-time feedback"""
        h, w = frame.shape[:2]
        
        # Main display area with transparency
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (w-10, 180), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        
        # Current word with enhanced styling
        word_text = f"Word: {self.current_word}"
        word_color = (0, 255, 255) if self.current_word else (128, 128, 128)
        cv2.putText(frame, word_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, word_color, 3)
        
        # Real-time word predictions
        if self.partial_word_predictions:
            prediction_text = f"Predictions: {' | '.join(self.partial_word_predictions)}"
            cv2.putText(frame, prediction_text, (20, 85), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Word confidence indicator
        if self.current_word:
            word_confidence = self.calculate_word_confidence()
            confidence_text = f"Confidence: {word_confidence:.2f}"
            confidence_color = (0, 255, 0) if word_confidence > 0.7 else (0, 165, 255) if word_confidence > 0.4 else (0, 0, 255)
            cv2.putText(frame, confidence_text, (20, 115), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, confidence_color, 2)
        
        # Recent words with better formatting
        if self.recognized_words:
            recent_words = " → ".join(self.recognized_words[-4:])
            cv2.putText(frame, f"Recent: {recent_words}", (20, 145), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Enhanced system status
        status_items = []
        if MEDIAPIPE_AVAILABLE:
            status_items.append(("MediaPipe", "✅", (0, 255, 0)))
        else:
            status_items.append(("MediaPipe", "❌", (0, 0, 255)))
        
        if self.letter_recognizer.model_available:
            status_items.append(("Model", "✅", (0, 255, 0)))
        else:
            status_items.append(("Model", "❌", (0, 0, 255)))
        
        if TTS_AVAILABLE:
            status_items.append(("TTS", "✅", (0, 255, 0)))
        else:
            status_items.append(("TTS", "❌", (0, 0, 255)))
        
        # Draw status items
        for i, (name, status, color) in enumerate(status_items):
            x_pos = 20 + i * 120
            cv2.putText(frame, f"{name}: {status}", (x_pos, 175), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Performance metrics
        avg_processing_time = np.mean(self.processing_times) if self.processing_times else 0
        cv2.putText(frame, f"FPS: {self.fps}", (w - 200, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Processing: {avg_processing_time:.1f}ms", (w - 200, 55), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Letters: {len(self.current_word)}", (w - 200, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f"Words: {len(self.recognized_words)}", (w - 200, 105), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Path length indicator
        if self.current_path:
            path_length = len(self.current_path)
            path_color = (0, 255, 0) if path_length >= self.MIN_PATH_LENGTH else (0, 165, 255)
            cv2.putText(frame, f"Path: {path_length}", (w - 200, 130), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, path_color, 1)
        
        # Enhanced instructions
        instructions = [
            "✋ Hold index finger up and write letters in air",
            "⌨️ SPACE: End letter | S: Speak | C: Clear | T: Toggle trail | ESC: Exit"
        ]
        
        for i, instruction in enumerate(instructions):
            cv2.putText(frame, instruction, (20, h - 60 + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        
        # Enhanced target words panel
        self.draw_enhanced_target_words_panel(frame)
    
    def draw_enhanced_target_words_panel(self, frame):
        """Enhanced target words panel with predictions and matching"""
        h, w = frame.shape[:2]
        
        # Panel dimensions
        panel_x, panel_y = w - 350, 10
        panel_w, panel_h = 330, 250
        
        # Draw panel background
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        
        # Title
        cv2.putText(frame, "Target Words:", (panel_x + 10, panel_y + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Show matching words first if current word exists
        words_to_show = []
        if self.current_word:
            # Show words that match current input
            matching_words = [word for word in self.word_corrector.target_words 
                            if word.startswith(self.current_word)]
            words_to_show.extend(matching_words[:12])
            
            # Fill remaining slots with other words
            remaining_slots = 18 - len(words_to_show)
            other_words = [word for word in self.word_corrector.target_words[:remaining_slots] 
                          if word not in words_to_show]
            words_to_show.extend(other_words)
        else:
            # Show first 18 words
            words_to_show = self.word_corrector.target_words[:18]
        
        # Display words in grid
        cols, rows = 3, 6
        
        for i, word in enumerate(words_to_show):
            if i >= 18:  # Limit display
                break
                
            row = i // cols
            col = i % cols
            
            x = panel_x + 15 + col * 100
            y = panel_y + 50 + row * 30
            
            # Determine color based on matching
            if self.current_word and word.startswith(self.current_word):
                if word == self.current_word:
                    color = (0, 255, 0)  # Exact match - green
                else:
                    color = (0, 255, 255)  # Partial match - cyan
            elif word in self.recognized_words[-3:]:
                color = (255, 0, 255)  # Recently recognized - magenta
            else:
                color = (200, 200, 200)  # Default - light gray
            
            # Draw word with appropriate styling
            font_scale = 0.5 if len(word) <= 3 else 0.4
            thickness = 2 if color != (200, 200, 200) else 1
            cv2.putText(frame, word, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
        
        # Show legend
        legend_y = panel_y + panel_h - 60
        cv2.putText(frame, "Legend:", (panel_x + 10, legend_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        legend_items = [
            ("Current", (0, 255, 0)),
            ("Match", (0, 255, 255)),
            ("Recent", (255, 0, 255))
        ]
        
        for i, (label, color) in enumerate(legend_items):
            x_pos = panel_x + 10 + i * 80
            cv2.putText(frame, label, (x_pos, legend_y + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    def process_letter_completion(self):
        """Enhanced letter processing with multiple candidates and smoothing"""
        if len(self.current_path) < self.MIN_PATH_LENGTH:
            return
        
        # Get letter recognition with confidence
        letter, confidence = self.letter_recognizer.recognize_letter(self.current_path)
        
        if letter and confidence > self.CONFIDENCE_THRESHOLD:
            # Store letter candidate
            self.letter_candidates.append((letter, confidence, time.time()))
            
            # Apply letter smoothing if enabled
            if self.letter_smoothing and len(self.letter_candidates) >= 2:
                # Check consistency with recent letters
                recent_letters = [l for l, c, t in self.letter_candidates[-3:]]
                if recent_letters.count(letter) >= 2:  # Majority vote
                    final_letter = letter
                else:
                    # Use highest confidence from recent candidates
                    best_candidate = max(self.letter_candidates[-3:], key=lambda x: x[1])
                    final_letter = best_candidate[0]
            else:
                final_letter = letter
            
            # Add letter to current word
            self.current_word += final_letter
            self.last_letter_time = time.time()
            
            # Update word predictions
            self.update_word_predictions()
            
            print(f"Letter: {final_letter} (confidence: {confidence:.3f})")
            
            # Auto-complete word if high confidence match found
            if self.auto_word_completion:
                self.check_auto_completion()
        else:
            print(f"Letter rejected: {letter} (confidence: {confidence:.3f})")
        
        # Clear path and reset counters
        self.current_path.clear()
        self.letter_pause_count = 0
    
    def process_word_completion(self):
        """Enhanced word completion with confidence scoring"""
        if not self.current_word:
            return
        
        # Get word correction with confidence
        corrected_word = self.word_corrector.correct_word(self.current_word)
        
        # Calculate word confidence based on letter confidences
        word_confidence = self.calculate_word_confidence()
        self.word_confidence_history.append(word_confidence)
        
        print(f"Word: {self.current_word} -> {corrected_word} (confidence: {word_confidence:.3f})")
        
        # Add to recognized words
        self.recognized_words.append(corrected_word)
        
        # Speak word in separate thread
        threading.Thread(target=self.tts.speak, args=(corrected_word,), daemon=True).start()
        
        # Reset state
        self.reset_word_state()
    
    def update_word_predictions(self):
        """Update real-time word predictions"""
        if self.current_word:
            suggestions = self.word_corrector.get_suggestions(self.current_word)
            self.partial_word_predictions = suggestions[:3]  # Top 3 predictions
    
    def check_auto_completion(self):
        """Check if word can be auto-completed with high confidence"""
        if len(self.current_word) >= 2:
            exact_matches = [word for word in self.word_corrector.target_words 
                           if word.startswith(self.current_word)]
            
            # Auto-complete if only one exact match and word is 3+ letters
            if len(exact_matches) == 1 and len(self.current_word) >= 3:
                completed_word = exact_matches[0]
                if len(completed_word) == len(self.current_word):
                    # Word is already complete
                    self.process_word_completion()
    
    def calculate_word_confidence(self):
        """Calculate overall word confidence from letter candidates"""
        if not self.letter_candidates:
            return 0.0
        
        # Get recent letter confidences
        recent_confidences = [c for l, c, t in self.letter_candidates[-len(self.current_word):]]
        
        if recent_confidences:
            return sum(recent_confidences) / len(recent_confidences)
        return 0.0
    
    def reset_word_state(self):
        """Reset all word-related state"""
        self.current_word = ""
        self.word_pause_count = 0
        self.current_path.clear()
        self.letter_candidates.clear()
        self.partial_word_predictions.clear()
    
    def run(self):
        """Optimized main application loop for real-time performance"""
        print("🚀 Starting Complete Air Writing System...")
        
        # Performance optimization variables
        frame_skip_counter = 0
        process_every_n_frames = 1  # Process every frame for maximum responsiveness
        
        while True:
            frame_start_time = time.time()
            
            ret, frame = self.cap.read()
            if not ret:
                print("❌ Failed to read from camera")
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Skip processing for performance if needed
            frame_skip_counter += 1
            if frame_skip_counter % process_every_n_frames != 0:
                cv2.imshow("Complete Air Writing System", frame)
                if cv2.waitKey(1) & 0xFF == 27:  # ESC
                    break
                continue
            
            # Process hand tracking with timing
            processing_start = time.time()
            fingertip, velocity, is_writing = self.hand_tracker.process_frame(frame)
            processing_time = (time.time() - processing_start) * 1000  # Convert to ms
            self.processing_times.append(processing_time)
            
            # Update FPS counter
            self.update_fps()
            
            # Enhanced writing logic with better state management
            current_time = time.time()
            
            if fingertip and is_writing:
                # Active writing detected
                self.current_path.append(fingertip)
                
                # Prevent path from becoming too long (memory optimization)
                if len(self.current_path) > self.MAX_PATH_LENGTH:
                    self.current_path = self.current_path[-self.MAX_PATH_LENGTH//2:]
                
                self.last_movement_time = current_time
                self.letter_pause_count = 0
                self.word_pause_count = 0
                
            else:
                # No active writing - handle pauses and completions
                if len(self.current_path) > 0:
                    self.letter_pause_count += 1
                
                # Letter completion with improved timing
                if (self.letter_pause_count >= self.LETTER_PAUSE_FRAMES or 
                    (len(self.current_path) >= self.MIN_PATH_LENGTH and 
                     current_time - self.last_movement_time > 1.0)):
                    self.process_letter_completion()
                
                # Word completion with multiple triggers
                time_since_last_movement = current_time - self.last_movement_time
                time_since_last_letter = current_time - self.last_letter_time
                
                if (time_since_last_movement > 2.5 or 
                    (self.current_word and time_since_last_letter > 4.0)):
                    self.word_pause_count += 1
                    
                    if (self.word_pause_count >= self.WORD_PAUSE_FRAMES or
                        time_since_last_movement > 5.0):
                        self.process_word_completion()
            
            # Draw enhanced UI
            self.draw_ui(frame)
            
            # Show frame with window management
            cv2.imshow("Complete Air Writing System", frame)
            
            # Enhanced keyboard input handling
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' '):  # Manual letter completion
                self.process_letter_completion()
            
            elif key == ord('s'):  # Speak current word
                if self.current_word:
                    corrected_word = self.word_corrector.correct_word(self.current_word)
                    self.recognized_words.append(corrected_word)
                    threading.Thread(target=self.tts.speak, args=(corrected_word,), daemon=True).start()
                    self.reset_word_state()
                    print(f"Spoken: {corrected_word}")
            
            elif key == ord('c'):  # Clear current word
                self.reset_word_state()
                self.hand_tracker.clear_trail()
                print("🧹 Cleared current word")
            
            elif key == ord('t'):  # Toggle trail visibility
                self.show_trail = not self.show_trail
                print(f"🎨 Trail {'enabled' if self.show_trail else 'disabled'}")
            
            elif key == ord('d'):  # Toggle debug info
                self.show_debug = not self.show_debug
                print(f"🐛 Debug info {'enabled' if self.show_debug else 'disabled'}")
            
            elif key == ord('a'):  # Toggle auto-completion
                self.auto_word_completion = not self.auto_word_completion
                print(f"🤖 Auto-completion {'enabled' if self.auto_word_completion else 'disabled'}")
            
            elif key == ord('r'):  # Reset session
                self.recognized_words.clear()
                self.reset_word_state()
                self.hand_tracker.clear_trail()
                print("🔄 Session reset")
            
            elif key == 27:  # ESC - Exit
                break
            
            # Performance monitoring
            frame_time = (time.time() - frame_start_time) * 1000
            if frame_time > 50:  # If frame takes more than 50ms, consider frame skipping
                process_every_n_frames = min(3, process_every_n_frames + 1)
            elif frame_time < 20:  # If frame is fast, process every frame
                process_every_n_frames = max(1, process_every_n_frames - 1)
        
        # Cleanup and session summary
        self.cleanup_and_summarize()
    
    def cleanup_and_summarize(self):
        """Cleanup resources and show session summary"""
        self.cap.release()
        cv2.destroyAllWindows()
        
        # Detailed session summary
        print(f"\n📊 SESSION SUMMARY")
        print("=" * 50)
        print(f"   Words recognized: {len(self.recognized_words)}")
        if self.recognized_words:
            print(f"   Words: {' → '.join(self.recognized_words)}")
        
        if self.word_confidence_history:
            avg_confidence = np.mean(self.word_confidence_history)
            print(f"   Average confidence: {avg_confidence:.3f}")
        
        if self.processing_times:
            avg_processing = np.mean(self.processing_times)
            print(f"   Average processing time: {avg_processing:.1f}ms")
        
        print(f"   Average FPS: {self.fps}")
        print("=" * 50)
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