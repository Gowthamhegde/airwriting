#!/usr/bin/env python3
"""
Enhanced Real-Time Air Writing Recognition System
Features:
- Accurate word detection with ensemble models
- Hand gesture controls (open/closed hand detection)
- Real-time letter recognition with confidence scoring
- Auto-correction and word completion
- Voice feedback
- Advanced trail visualization
- Performance optimization
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import json
import os
import threading
import argparse
from collections import deque
from datetime import datetime
from pathlib import Path

# Try imports with fallbacks
try:
    from tensorflow.keras.models import load_model
    import tensorflow as tf
    MODEL_AVAILABLE = True
except ImportError:
    MODEL_AVAILABLE = False
    print("⚠️  TensorFlow not available - using demo recognition")

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False
    print("⚠️  TextBlob not available - basic correction only")

try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    print("⚠️  Text-to-speech not available")

class EnhancedHandTracker:
    """Enhanced hand tracker with gesture recognition and writing detection"""
    
    def __init__(self, max_hands=1, trail_length=200):
        self.hands_module = mp.solutions.hands
        self.hands = self.hands_module.Hands(
            max_num_hands=max_hands,
            min_detection_confidence=0.8,
            min_tracking_confidence=0.7,
            model_complexity=1
        )
        self.drawing = mp.solutions.drawing_utils
        self.drawing_styles = mp.solutions.drawing_styles
        
        # Tracking state
        self.fingertip_trail = deque(maxlen=trail_length)
        self.velocity_history = deque(maxlen=10)
        self.hand_present = False
        self.writing_mode = False
        self.hand_open = False
        
        # Smoothing parameters
        self.alpha = 0.3  # EMA smoothing factor
        self.prev_smoothed_pos = None
        self.velocity = 0
        
        # Gesture detection
        self.gesture_history = deque(maxlen=15)
        self.last_gesture_change = time.time()
        self.gesture_stability_threshold = 0.5  # seconds
        
        print("✅ Enhanced hand tracker initialized")
    
    def detect_hand_openness(self, hand_landmarks):
        """
        Detect if hand is open or closed
        Returns: (is_open, confidence)
        """
        # Get fingertip and joint positions
        fingertips = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky
        joints = [3, 6, 10, 14, 18]      # Corresponding joints
        
        extended_fingers = 0
        finger_confidences = []
        
        for tip_idx, joint_idx in zip(fingertips, joints):
            tip = hand_landmarks.landmark[tip_idx]
            joint = hand_landmarks.landmark[joint_idx]
            
            # For thumb, check x-axis extension (different orientation)
            if tip_idx == 4:
                is_extended = abs(tip.x - joint.x) > 0.04
            else:
                # For other fingers, check y-axis extension
                is_extended = tip.y < joint.y - 0.02
            
            if is_extended:
                extended_fingers += 1
                finger_confidences.append(1.0)
            else:
                finger_confidences.append(0.0)
        
        # Calculate confidence based on finger extension
        confidence = np.mean(finger_confidences)
        is_open = extended_fingers >= 3  # At least 3 fingers extended = open hand
        
        return is_open, confidence
    
    def detect_writing_gesture(self, hand_landmarks):
        """
        Detect writing gesture (index finger extended, others curled)
        Returns: (is_writing, confidence)
        """
        # Index finger positions
        index_tip = hand_landmarks.landmark[8]
        index_pip = hand_landmarks.landmark[6]
        index_mcp = hand_landmarks.landmark[5]
        
        # Other finger positions
        middle_tip = hand_landmarks.landmark[12]
        middle_pip = hand_landmarks.landmark[10]
        ring_tip = hand_landmarks.landmark[16]
        ring_pip = hand_landmarks.landmark[14]
        pinky_tip = hand_landmarks.landmark[20]
        pinky_pip = hand_landmarks.landmark[18]
        
        # Check index finger extension
        index_extended = index_tip.y < index_pip.y < index_mcp.y
        
        # Check other fingers are curled
        middle_curled = middle_tip.y > middle_pip.y
        ring_curled = ring_tip.y > ring_pip.y
        pinky_curled = pinky_tip.y > pinky_pip.y
        
        # Calculate confidence
        confidence = 0.0
        if index_extended:
            confidence += 0.4
        if middle_curled:
            confidence += 0.2
        if ring_curled:
            confidence += 0.2
        if pinky_curled:
            confidence += 0.2
        
        is_writing = confidence > 0.6
        return is_writing, confidence
    
    def smooth_position(self, current_pos):
        """Apply exponential moving average smoothing"""
        if self.prev_smoothed_pos is None:
            smoothed_pos = current_pos
        else:
            smoothed_pos = (
                int(self.alpha * current_pos[0] + (1 - self.alpha) * self.prev_smoothed_pos[0]),
                int(self.alpha * current_pos[1] + (1 - self.alpha) * self.prev_smoothed_pos[1])
            )
        
        self.prev_smoothed_pos = smoothed_pos
        return smoothed_pos
    
    def calculate_velocity(self, current_pos):
        """Calculate movement velocity"""
        if len(self.fingertip_trail) > 0:
            prev_pos = self.fingertip_trail[-1]
            dx = current_pos[0] - prev_pos[0]
            dy = current_pos[1] - prev_pos[1]
            velocity = np.sqrt(dx*dx + dy*dy)
            
            self.velocity_history.append(velocity)
            self.velocity = np.mean(list(self.velocity_history))
        else:
            self.velocity = 0
    
    def update_gesture_history(self, is_open, is_writing):
        """Update gesture history for stability"""
        current_time = time.time()
        gesture_state = {
            'open': is_open,
            'writing': is_writing,
            'time': current_time
        }
        
        self.gesture_history.append(gesture_state)
        
        # Check for stable gesture
        if len(self.gesture_history) >= 5:
            recent_gestures = list(self.gesture_history)[-5:]
            
            # Check if gesture has been stable
            open_votes = sum(1 for g in recent_gestures if g['open'])
            writing_votes = sum(1 for g in recent_gestures if g['writing'])
            
            stable_open = open_votes >= 3
            stable_writing = writing_votes >= 3
            
            # Update state only if stable
            if current_time - self.last_gesture_change > self.gesture_stability_threshold:
                if stable_open != self.hand_open or stable_writing != self.writing_mode:
                    self.hand_open = stable_open
                    self.writing_mode = stable_writing
                    self.last_gesture_change = current_time
                    
                    if stable_open and not stable_writing:
                        print("✋ Hand detected: OPEN - Tracking paused")
                    elif stable_writing:
                        print("✍️ Hand detected: WRITING - Tracking active")
                    elif not stable_open:
                        print("✊ Hand detected: CLOSED - Tracking paused")
    
    def process_frame(self, frame):
        """Process frame and detect hand gestures"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        fingertip = None
        
        if results.multi_hand_landmarks:
            self.hand_present = True
            
            for hand_landmarks in results.multi_hand_landmarks:
                h, w, _ = frame.shape
                
                # Detect hand openness and writing gesture
                is_open, open_confidence = self.detect_hand_openness(hand_landmarks)
                is_writing, writing_confidence = self.detect_writing_gesture(hand_landmarks)
                
                # Update gesture history for stability
                self.update_gesture_history(is_open, is_writing)
                
                # Get index fingertip position
                index_tip = hand_landmarks.landmark[8]
                raw_pos = (int(index_tip.x * w), int(index_tip.y * h))
                
                # Apply smoothing
                smoothed_pos = self.smooth_position(raw_pos)
                
                # Calculate velocity
                self.calculate_velocity(smoothed_pos)
                
                # Only track fingertip if in writing mode and hand is not fully open
                if self.writing_mode and not self.hand_open:
                    fingertip = smoothed_pos
                    self.fingertip_trail.append(smoothed_pos)
                
                # Draw hand landmarks
                self.drawing.draw_landmarks(
                    frame, hand_landmarks, self.hands_module.HAND_CONNECTIONS,
                    self.drawing_styles.get_default_hand_landmarks_style(),
                    self.drawing_styles.get_default_hand_connections_style()
                )
                
                # Draw status information
                self.draw_hand_status(frame, is_open, is_writing, open_confidence, writing_confidence)
                
                # Draw fingertip indicator
                if fingertip:
                    self.draw_fingertip_indicator(frame, fingertip)
        
        else:
            # No hand detected
            self.hand_present = False
            self.writing_mode = False
            self.hand_open = False
            self.fingertip_trail.clear()
            self.velocity_history.clear()
            self.prev_smoothed_pos = None
            self.velocity = 0
        
        return fingertip, frame
    
    def draw_hand_status(self, frame, is_open, is_writing, open_conf, writing_conf):
        """Draw hand status information"""
        status_y = 30
        
        # Hand presence
        hand_status = "✅ Hand Detected" if self.hand_present else "❌ No Hand"
        hand_color = (0, 255, 0) if self.hand_present else (0, 0, 255)
        cv2.putText(frame, hand_status, (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, hand_color, 2)
        
        if self.hand_present:
            # Hand openness
            open_status = f"Hand: {'OPEN' if self.hand_open else 'CLOSED'} ({open_conf:.2f})"
            open_color = (255, 255, 0) if self.hand_open else (255, 0, 255)
            cv2.putText(frame, open_status, (10, status_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, open_color, 2)
            
            # Writing mode
            writing_status = f"Mode: {'WRITING' if self.writing_mode else 'IDLE'} ({writing_conf:.2f})"
            writing_color = (0, 255, 255) if self.writing_mode else (128, 128, 128)
            cv2.putText(frame, writing_status, (10, status_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, writing_color, 2)
            
            # Tracking status
            tracking_status = "Tracking: " + ("ACTIVE" if self.is_tracking_active() else "PAUSED")
            tracking_color = (0, 255, 0) if self.is_tracking_active() else (0, 0, 255)
            cv2.putText(frame, tracking_status, (10, status_y + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, tracking_color, 2)
    
    def draw_fingertip_indicator(self, frame, fingertip):
        """Draw enhanced fingertip indicator"""
        # Pulsing effect based on velocity
        pulse_factor = min(1.0, self.velocity / 10.0)
        base_radius = 8
        pulse_radius = int(base_radius + pulse_factor * 4)
        
        # Color based on movement
        if self.velocity > 5:
            color = (0, 0, 255)  # Red for fast movement
        elif self.velocity > 2:
            color = (0, 255, 255)  # Yellow for medium movement
        else:
            color = (0, 255, 0)  # Green for slow/stationary
        
        # Draw fingertip
        cv2.circle(frame, fingertip, pulse_radius, color, -1)
        cv2.circle(frame, fingertip, pulse_radius + 2, (255, 255, 255), 2)
        
        # Draw velocity indicator
        if self.velocity > 1:
            velocity_text = f"{self.velocity:.1f}"
            cv2.putText(frame, velocity_text, (fingertip[0] + 15, fingertip[1] - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    def draw_trail(self, frame):
        """Draw enhanced trail with gradient effects"""
        if len(self.fingertip_trail) < 2:
            return
        
        trail_points = list(self.fingertip_trail)
        
        for i in range(1, len(trail_points)):
            # Calculate progress for gradient
            progress = i / len(trail_points)
            
            # Color gradient from blue to red
            blue = int(255 * (1 - progress))
            red = int(255 * progress)
            color = (blue, 0, red)
            
            # Thickness based on progress and velocity
            thickness = max(1, int(2 + progress * 4))
            
            cv2.line(frame, trail_points[i-1], trail_points[i], color, thickness)
    
    def get_trail_path(self):
        """Get current trail as list of points"""
        return list(self.fingertip_trail)
    
    def clear_trail(self):
        """Clear the current trail"""
        self.fingertip_trail.clear()
    
    def is_tracking_active(self):
        """Check if tracking is currently active"""
        return self.hand_present and self.writing_mode and not self.hand_open

class AdvancedLetterRecognizer:
    """Advanced letter recognizer with ensemble models and confidence scoring"""
    
    def __init__(self, model_paths=None):
        self.models = []
        self.model_available = False
        self.processing_times = deque(maxlen=30)
        
        # Load models
        self.load_models(model_paths)
        
        print(f"🧠 Advanced letter recognizer initialized (Models: {len(self.models)})")
    
    def load_models(self, model_paths=None):
        """Load trained models"""
        if not MODEL_AVAILABLE:
            print("⚠️  TensorFlow not available - using demo mode")
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
                    
                    self.models.append(model)
                    print(f"✅ Loaded model: {os.path.basename(path)}")
                    
                except Exception as e:
                    print(f"⚠️  Failed to load {path}: {e}")
        
        if self.models:
            self.model_available = True
    
    def create_letter_image(self, path, img_size=28):
        """Create enhanced letter image from path"""
        if len(path) < 2:
            return np.ones((img_size, img_size), dtype=np.uint8) * 255
        
        # Create blank image
        img = np.ones((img_size, img_size), dtype=np.uint8) * 255
        
        try:
            # Convert path to numpy array
            path_array = np.array(path)
            
            # Get bounding box
            min_x, min_y = np.min(path_array, axis=0)
            max_x, max_y = np.max(path_array, axis=0)
            
            width = max_x - min_x
            height = max_y - min_y
            
            if width <= 0 or height <= 0:
                return img
            
            # Scale to fit image with padding
            padding = 4
            available_size = img_size - 2 * padding
            scale = min(available_size / width, available_size / height)
            
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
            
            # Draw lines with variable thickness
            for i in range(1, len(scaled_path)):
                thickness = 2
                cv2.line(img, scaled_path[i-1], scaled_path[i], (0), thickness)
            
            # Apply slight blur for anti-aliasing
            img = cv2.GaussianBlur(img, (3, 3), 0.5)
            
            return img
            
        except Exception as e:
            print(f"⚠️ Image creation error: {e}")
            return img
    
    def recognize_letter(self, path):
        """Recognize letter with ensemble prediction"""
        if len(path) < 5:
            return None, 0.0
        
        start_time = time.time()
        
        try:
            if self.model_available and self.models:
                # Create image
                img = self.create_letter_image(path)
                img_normalized = img.reshape(1, 28, 28, 1).astype(np.float32) / 255.0
                
                # Ensemble prediction
                if len(self.models) > 1:
                    predictions = []
                    for model in self.models:
                        try:
                            pred = model.predict(img_normalized, verbose=0)[0]
                            predictions.append(pred)
                        except Exception as e:
                            print(f"⚠️ Model prediction error: {e}")
                            continue
                    
                    if predictions:
                        # Average predictions
                        final_prediction = np.mean(predictions, axis=0)
                    else:
                        return self.demo_recognition()
                else:
                    try:
                        final_prediction = self.models[0].predict(img_normalized, verbose=0)[0]
                    except Exception as e:
                        print(f"⚠️ Model prediction error: {e}")
                        return self.demo_recognition()
                
                # Get letter and confidence
                letter_idx = np.argmax(final_prediction)
                confidence = final_prediction[letter_idx]
                letter = chr(letter_idx + ord('A'))
                
                # Enhanced confidence calculation
                sorted_probs = np.sort(final_prediction)[::-1]
                top_prob = sorted_probs[0]
                second_prob = sorted_probs[1] if len(sorted_probs) > 1 else 0
                
                # Confidence boost based on separation
                separation = top_prob - second_prob
                adjusted_confidence = min(1.0, confidence + separation * 0.2)
                
                # Track processing time
                processing_time = (time.time() - start_time) * 1000
                self.processing_times.append(processing_time)
                
                return letter, adjusted_confidence
            else:
                return self.demo_recognition()
                
        except Exception as e:
            print(f"❌ Recognition error: {e}")
            return self.demo_recognition()
    
    def demo_recognition(self):
        """Demo recognition for when models aren't available"""
        letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        letter = np.random.choice(list(letters))
        confidence = np.random.uniform(0.5, 0.9)
        return letter, confidence
    
    def get_average_processing_time(self):
        """Get average processing time"""
        return np.mean(self.processing_times) if self.processing_times else 0.0

class SmartWordCorrector:
    """Smart word corrector with dictionary and auto-correction"""
    
    def __init__(self, dictionary_path=None):
        self.target_words = []
        self.word_frequencies = {}
        self.load_dictionary(dictionary_path)
        
        print(f"📚 Smart word corrector initialized with {len(self.target_words)} words")
    
    def load_dictionary(self, dictionary_path=None):
        """Load dictionary from file or use default"""
        if dictionary_path and os.path.exists(dictionary_path):
            try:
                with open(dictionary_path, 'r') as f:
                    data = json.load(f)
                    self.target_words = data.get('target_words', [])
                    self.word_frequencies = data.get('word_frequencies', {})
                print(f"✅ Loaded dictionary: {dictionary_path}")
                return
            except Exception as e:
                print(f"⚠️  Failed to load dictionary: {e}")
        
        # Default dictionary
        self.target_words = [
            # Common 3-letter words
            'THE', 'AND', 'FOR', 'ARE', 'BUT', 'NOT', 'YOU', 'ALL', 'CAN', 'HER', 'WAS', 'ONE', 'OUR',
            'HAD', 'HIS', 'HAS', 'SHE', 'TWO', 'WAY', 'WHO', 'BOY', 'DID', 'ITS', 'LET', 'PUT',
            'SAY', 'TOO', 'OLD', 'ANY', 'MAY', 'NEW', 'TRY', 'ASK', 'MAN', 'DAY', 'GET', 'USE',
            'NOW', 'AIR', 'END', 'WHY', 'HOW', 'OUT', 'SEE', 'HIM', 'OIL', 'SUN', 'ICE', 'TOP',
            
            # Animals and objects
            'CAT', 'DOG', 'BAT', 'RAT', 'PIG', 'COW', 'BEE', 'ANT', 'FOX', 'OWL', 'ELF',
            'HAT', 'MAT', 'BAG', 'BOX', 'CUP', 'PEN', 'KEY', 'CAR', 'BUS', 'BED', 'EGG',
            
            # Actions
            'RUN', 'SIT', 'EAT', 'SEE', 'HIT', 'CUT', 'DIG', 'FLY', 'WIN', 'TRY', 'BUY',
            'PAY', 'SAY', 'LAY', 'WAG', 'HOP', 'MOP', 'POP', 'COP', 'SOP', 'JIG', 'RIG',
            
            # Common 4-letter words
            'THAT', 'WITH', 'HAVE', 'THIS', 'WILL', 'YOUR', 'FROM', 'THEY', 'KNOW', 'WANT',
            'BEEN', 'GOOD', 'MUCH', 'SOME', 'TIME', 'VERY', 'WHEN', 'COME', 'HERE', 'JUST',
            'LIKE', 'LONG', 'MAKE', 'MANY', 'OVER', 'SUCH', 'TAKE', 'THAN', 'THEM', 'WELL',
            'WORK', 'CALL', 'CAME', 'EACH', 'EVEN', 'FIND', 'GIVE', 'HAND', 'HIGH', 'KEEP',
            'LAST', 'LEFT', 'LIFE', 'LIVE', 'LOOK', 'MADE', 'MOST', 'MOVE', 'MUST', 'NAME',
            'NEED', 'NEXT', 'OPEN', 'PART', 'PLAY', 'SAID', 'SAME', 'SEEM', 'SHOW', 'SIDE',
            'TELL', 'TURN', 'USED', 'WAYS', 'WEEK', 'WENT', 'WERE', 'WHAT', 'WORD', 'YEAR',
            'BACK', 'BOOK', 'FACE', 'FACT', 'FEEL', 'FIRE', 'FOOD', 'FORM', 'FOUR', 'FREE',
            'FULL', 'GAME', 'GIRL', 'GOES', 'HELP', 'HOME', 'HOPE', 'HOUR', 'IDEA', 'KIND',
            'LAND', 'LATE', 'LINE', 'LIST', 'LOVE', 'MIND', 'NICE', 'ONLY', 'PLAN', 'REAL',
            'ROOM', 'SAVE', 'SEND', 'SOON', 'STOP', 'SURE', 'TALK', 'TEAM', 'TREE', 'TRUE',
            'TYPE', 'VIEW', 'WALK', 'WALL', 'WEAR', 'WIFE', 'WIND',
            
            # Common 5-letter words
            'ABOUT', 'AFTER', 'AGAIN', 'BEING', 'COULD', 'EVERY', 'FIRST', 'FOUND', 'GREAT',
            'GROUP', 'HOUSE', 'LARGE', 'NEVER', 'OTHER', 'PLACE', 'RIGHT', 'SHALL', 'SMALL',
            'SOUND', 'STILL', 'THEIR', 'THERE', 'THESE', 'THING', 'THINK', 'THREE', 'UNDER',
            'WATER', 'WHERE', 'WHICH', 'WHILE', 'WORLD', 'WOULD', 'WRITE', 'YOUNG'
        ]
        
        # Create frequency map
        self.word_frequencies = {}
        for i, word in enumerate(self.target_words):
            freq = 1.0 - (i * 0.005)
            self.word_frequencies[word] = max(0.1, freq)
    
    def levenshtein_distance(self, s1, s2):
        """Calculate Levenshtein distance"""
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
        """Correct word using dictionary"""
        if not word:
            return "", 0.0
        
        word = word.upper().strip()
        
        # Direct match
        if word in self.target_words:
            return word, 1.0
        
        # Find best match
        best_word = word
        best_score = 0.0
        
        for candidate in self.target_words:
            # Calculate similarity
            distance = self.levenshtein_distance(word, candidate)
            max_len = max(len(word), len(candidate))
            similarity = 1.0 - (distance / max_len) if max_len > 0 else 0.0
            
            # Frequency bonus
            frequency_bonus = self.word_frequencies.get(candidate, 0.5) * 0.1
            
            # Length penalty for very different lengths
            length_penalty = abs(len(word) - len(candidate)) * 0.05
            
            final_score = similarity + frequency_bonus - length_penalty
            final_score = max(0.0, min(1.0, final_score))
            
            if final_score > best_score:
                best_score = final_score
                best_word = candidate
        
        # Try TextBlob if available and no good match
        if TEXTBLOB_AVAILABLE and best_score < 0.5:
            try:
                blob = TextBlob(word.lower())
                corrected = str(blob.correct()).upper()
                if corrected in self.target_words:
                    tb_distance = self.levenshtein_distance(word, corrected)
                    tb_score = max(0.6, 1.0 - (tb_distance / max(len(word), len(corrected))))
                    if tb_score > best_score:
                        return corrected, tb_score
            except Exception as e:
                print(f"⚠️ TextBlob correction error: {e}")
        
        return best_word, best_score
    
    def get_suggestions(self, partial_word, max_suggestions=3):
        """Get word completion suggestions"""
        if not partial_word:
            # Return most frequent words
            frequent_words = sorted(self.target_words, 
                                  key=lambda w: self.word_frequencies.get(w, 0), 
                                  reverse=True)
            return frequent_words[:max_suggestions]
        
        partial = partial_word.upper()
        suggestions = []
        
        # Find words that start with partial input
        for word in self.target_words:
            if word.startswith(partial):
                frequency = self.word_frequencies.get(word, 0.5)
                suggestions.append((word, frequency))
        
        # Sort by frequency and return top suggestions
        suggestions.sort(key=lambda x: x[1], reverse=True)
        return [word for word, freq in suggestions[:max_suggestions]]

class VoiceFeedback:
    """Voice feedback system with text-to-speech"""
    
    def __init__(self):
        self.tts_available = TTS_AVAILABLE
        self.engine = None
        self.speaking = False
        self.recent_words = deque(maxlen=5)
        self.last_speech_time = 0
        self.min_speech_interval = 1.0
        
        if TTS_AVAILABLE:
            try:
                self.engine = pyttsx3.init()
                self.configure_engine()
                print("🔊 Voice feedback initialized")
            except Exception as e:
                print(f"⚠️  TTS initialization failed: {e}")
                self.tts_available = False
    
    def configure_engine(self):
        """Configure TTS engine"""
        if not self.engine:
            return
        
        try:
            self.engine.setProperty('rate', 180)
            self.engine.setProperty('volume', 0.9)
            
            # Set voice
            voices = self.engine.getProperty('voices')
            if voices:
                for voice in voices:
                    if voice.name and ('female' in voice.name.lower() or 
                                     'zira' in voice.name.lower()):
                        self.engine.setProperty('voice', voice.id)
                        break
        except Exception as e:
            print(f"⚠️  TTS configuration error: {e}")
    
    def should_speak(self, word):
        """Check if word should be spoken"""
        current_time = time.time()
        
        if current_time - self.last_speech_time < self.min_speech_interval:
            return False
        
        if word in self.recent_words:
            return False
        
        return True
    
    def speak_word(self, word):
        """Speak word with threading"""
        if not self.tts_available or not word:
            print(f"🔊 Would speak: {word}")
            return
        
        word = word.strip().upper()
        
        if not self.should_speak(word):
            return
        
        def speak_thread():
            try:
                self.speaking = True
                print(f"🔊 Speaking: {word}")
                
                if self.engine:
                    self.engine.say(word)
                    self.engine.runAndWait()
                
                self.recent_words.append(word)
                self.last_speech_time = time.time()
                
            except Exception as e:
                print(f"❌ Speech error: {e}")
            finally:
                self.speaking = False
        
        speech_thread = threading.Thread(target=speak_thread, daemon=True)
        speech_thread.start()
    
    def is_speaking(self):
        """Check if currently speaking"""
        return self.speaking

class EnhancedAirWritingSystem:
    """Enhanced real-time air writing recognition system"""
    
    def __init__(self, model_paths=None, dictionary_path=None):
        print("🚀 Initializing Enhanced Air Writing System...")
        
        # Initialize components
        self.hand_tracker = EnhancedHandTracker()
        self.letter_recognizer = AdvancedLetterRecognizer(model_paths)
        self.word_corrector = SmartWordCorrector(dictionary_path)
        self.voice_feedback = VoiceFeedback()
        
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
        self.word_suggestions = []
        
        # Timing parameters
        self.min_path_length = 8
        self.letter_pause_threshold = 1.0  # seconds
        self.word_pause_threshold = 2.0    # seconds
        self.last_movement_time = time.time()
        
        # Performance tracking
        self.fps = 0
        self.frame_count = 0
        self.fps_start_time = time.time()
        
        # Display settings
        self.show_trail = True
        self.show_debug = False
        
        # Output logging
        self.output_file = "enhanced_airwriting_log.txt"
        
        print("✅ Enhanced Air Writing System initialized")
        self.print_instructions()
    
    def print_instructions(self):
        """Print usage instructions"""
        print("\n" + "="*80)
        print("🖐️  ENHANCED REAL-TIME AIR WRITING SYSTEM")
        print("="*80)
        print("📋 Instructions:")
        print("   • Hold up your INDEX FINGER (other fingers curled) to write")
        print("   • OPEN your hand (all fingers extended) to pause tracking")
        print("   • CLOSE your hand (make a fist) to pause tracking")
        print("   • Write letters clearly in the air")
        print("   • System auto-detects letter and word completion")
        print("   • Words are auto-corrected and spoken")
        print("\n⌨️  Controls:")
        print("   SPACE    - Force complete current letter")
        print("   ENTER    - Force complete current word")
        print("   C        - Clear current word")
        print("   R        - Reset everything")
        print("   T        - Toggle trail display")
        print("   D        - Toggle debug information")
        print("   V        - Test voice feedback")
        print("   ESC      - Exit system")
        print("="*80 + "\n")
    
    def update_fps(self):
        """Update FPS counter"""
        self.frame_count += 1
        current_time = time.time()
        
        if current_time - self.fps_start_time >= 1.0:
            self.fps = self.frame_count
            self.frame_count = 0
            self.fps_start_time = current_time
    
    def process_letter_completion(self):
        """Process letter completion"""
        if len(self.current_path) < self.min_path_length:
            return
        
        # Recognize letter
        letter, confidence = self.letter_recognizer.recognize_letter(self.current_path)
        
        # Adaptive confidence threshold
        min_confidence = 0.4
        if len(self.current_word) > 0:
            min_confidence = 0.3  # Lower threshold for subsequent letters
        
        if letter and confidence > min_confidence:
            self.current_word += letter
            print(f"✅ Letter: {letter} (confidence: {confidence:.3f})")
            
            # Update suggestions
            self.word_suggestions = self.word_corrector.get_suggestions(self.current_word)
            
            # Show auto-correction preview
            if len(self.current_word) >= 2:
                corrected_word, correction_confidence = self.word_corrector.correct_word(self.current_word)
                if corrected_word != self.current_word and correction_confidence > 0.6:
                    print(f"💡 Auto-correction preview: {self.current_word} → {corrected_word}")
        else:
            print(f"❌ Letter rejected: {letter} (confidence: {confidence:.3f}, needed: {min_confidence:.3f})")
        
        # Clear path
        self.current_path.clear()
        self.hand_tracker.clear_trail()
    
    def process_word_completion(self):
        """Process word completion"""
        if not self.current_word:
            return
        
        # Auto-correct word
        corrected_word, correction_confidence = self.word_corrector.correct_word(self.current_word)
        
        if corrected_word != self.current_word:
            print(f"🔄 Auto-correcting: {self.current_word} → {corrected_word} (confidence: {correction_confidence:.3f})")
        else:
            print(f"✅ Word recognized: {corrected_word}")
        
        # Add to recognized words
        self.recognized_words.append(corrected_word)
        
        # Log word
        self.log_word(corrected_word)
        
        # Speak word
        self.voice_feedback.speak_word(corrected_word)
        
        # Reset state
        self.reset_word_state()
    
    def reset_word_state(self):
        """Reset word state"""
        self.current_word = ""
        self.word_suggestions = []
        self.current_path.clear()
        self.hand_tracker.clear_trail()
    
    def log_word(self, word):
        """Log word to file"""
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_entry = f"{timestamp}: {word}\n"
            
            with open(self.output_file, "a", encoding="utf-8") as f:
                f.write(log_entry)
            
            print(f"📝 Logged: {word}")
        except Exception as e:
            print(f"⚠️ Logging error: {e}")
    
    def draw_ui(self, frame):
        """Draw user interface"""
        h, w = frame.shape[:2]
        
        # Create overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 100), (w-10, 250), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        
        # Current word
        word_display = self.current_word if self.current_word else "[Writing...]"
        word_color = (0, 255, 255) if self.current_word else (128, 128, 128)
        cv2.putText(frame, f"Current Word: {word_display}", (20, 130), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, word_color, 2)
        
        # Word suggestions
        if self.word_suggestions:
            suggestions_text = f"Suggestions: {' | '.join(self.word_suggestions[:3])}"
            cv2.putText(frame, suggestions_text, (20, 160), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Recent words
        if self.recognized_words:
            recent_words = " → ".join(self.recognized_words[-3:])
            cv2.putText(frame, f"Recent: {recent_words}", (20, 190), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Performance info
        cv2.putText(frame, f"FPS: {self.fps}", (20, 220), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        avg_proc_time = self.letter_recognizer.get_average_processing_time()
        cv2.putText(frame, f"Processing: {avg_proc_time:.1f}ms", (100, 220), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.putText(frame, f"Path Length: {len(self.current_path)}", (250, 220), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Instructions
        instructions = [
            "✋ INDEX finger up = Write | OPEN hand = Pause | CLOSED hand = Pause",
            "⌨️ SPACE: Complete letter | ENTER: Complete word | C: Clear | ESC: Exit"
        ]
        
        for i, instruction in enumerate(instructions):
            cv2.putText(frame, instruction, (20, h - 50 + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    
    def handle_keyboard_input(self, key):
        """Handle keyboard input"""
        if key == ord(' '):  # SPACE
            self.process_letter_completion()
        elif key == 13:  # ENTER
            if self.current_word:
                self.process_word_completion()
        elif key == ord('c') or key == ord('C'):
            self.reset_word_state()
            print("🧹 Cleared current word")
        elif key == ord('r') or key == ord('R'):
            self.recognized_words.clear()
            self.reset_word_state()
            print("🔄 Reset everything")
        elif key == ord('t') or key == ord('T'):
            self.show_trail = not self.show_trail
            print(f"🎨 Trail: {'ON' if self.show_trail else 'OFF'}")
        elif key == ord('d') or key == ord('D'):
            self.show_debug = not self.show_debug
            print(f"🐛 Debug: {'ON' if self.show_debug else 'OFF'}")
        elif key == ord('v') or key == ord('V'):
            print("🧪 Testing voice feedback...")
            self.voice_feedback.speak_word("TEST")
        elif key == 27:  # ESC
            return False
        
        return True
    
    def run(self):
        """Main application loop"""
        print("🚀 Starting Enhanced Air Writing System...")
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("❌ Failed to read frame")
                    break
                
                # Flip frame for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Process hand tracking
                fingertip, frame = self.hand_tracker.process_frame(frame)
                
                # Update FPS
                self.update_fps()
                
                # Handle writing logic
                current_time = time.time()
                
                if fingertip and self.hand_tracker.is_tracking_active():
                    # Active writing
                    self.current_path.append(fingertip)
                    self.last_movement_time = current_time
                else:
                    # Handle pauses
                    time_since_movement = current_time - self.last_movement_time
                    
                    # Letter completion
                    if (len(self.current_path) >= self.min_path_length and 
                        time_since_movement > self.letter_pause_threshold):
                        self.process_letter_completion()
                    
                    # Word completion
                    if (self.current_word and 
                        time_since_movement > self.word_pause_threshold):
                        print(f"⏰ Auto-completing word after {time_since_movement:.1f}s pause")
                        self.process_word_completion()
                
                # Draw trail
                if self.show_trail:
                    self.hand_tracker.draw_trail(frame)
                
                # Draw UI
                self.draw_ui(frame)
                
                # Show frame
                cv2.imshow('Enhanced Air Writing System', frame)
                
                # Handle input
                key = cv2.waitKey(1) & 0xFF
                if not self.handle_keyboard_input(key):
                    break
        
        except KeyboardInterrupt:
            print("\n👋 System interrupted by user")
        except Exception as e:
            print(f"❌ System error: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Cleanup resources"""
        print("🧹 Cleaning up...")
        
        if hasattr(self, 'cap') and self.cap:
            self.cap.release()
        
        cv2.destroyAllWindows()
        
        print("✅ Cleanup complete")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Enhanced Air Writing System")
    parser.add_argument('--models', nargs='+', help='Paths to model files')
    parser.add_argument('--dictionary', type=str, help='Path to dictionary file')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    print("🖐️  ENHANCED REAL-TIME AIR WRITING SYSTEM")
    print("=" * 60)
    print("🎯 Advanced hand gesture recognition")
    print("🧠 Ensemble model letter recognition")
    print("📚 Smart word auto-correction")
    print("🔊 Voice feedback")
    print("=" * 60)
    
    try:
        system = EnhancedAirWritingSystem(
            model_paths=args.models,
            dictionary_path=args.dictionary
        )
        system.run()
        
    except KeyboardInterrupt:
        print("\n👋 System interrupted by user")
    except Exception as e:
        print(f"❌ System error: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
    finally:
        print("🏁 Enhanced Air Writing System finished!")

if __name__ == "__main__":
    main()