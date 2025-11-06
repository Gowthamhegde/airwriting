#!/usr/bin/env python3
"""
Ultimate Air Writing Recognition System
Combines all advanced features:
- Precise hand gesture detection (open/closed hand control)
- Ensemble letter recognition with multiple models
- Smart word correction and completion
- Real-time performance optimization
- Voice feedback and logging
"""

import cv2
import numpy as np
import time
import json
import os
import threading
import argparse
from collections import deque
from datetime import datetime
from pathlib import Path

# Import our custom modules
from modules.advanced_hand_detection import AdvancedHandDetector, HandState, HandGestureController
from modules.ensemble_letter_recognition import EnsembleLetterRecognizer

# Try imports with fallbacks
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

class UltimateWordCorrector:
    """Ultimate word corrector with advanced algorithms"""
    
    def __init__(self, dictionary_path=None):
        self.target_words = []
        self.word_frequencies = {}
        self.word_set = set()
        self.load_dictionary(dictionary_path)
        
        print(f"📚 Ultimate word corrector initialized with {len(self.target_words)} words")
    
    def load_dictionary(self, dictionary_path=None):
        """Load comprehensive dictionary"""
        if dictionary_path and os.path.exists(dictionary_path):
            try:
                with open(dictionary_path, 'r') as f:
                    data = json.load(f)
                    self.target_words = data.get('target_words', [])
                    self.word_frequencies = data.get('word_frequencies', {})
                print(f"✅ Loaded dictionary: {dictionary_path}")
                self._finalize_setup()
                return
            except Exception as e:
                print(f"⚠️  Failed to load dictionary: {e}")
        
        # Comprehensive default dictionary
        self.target_words = [
            # High frequency words
            'THE', 'AND', 'FOR', 'ARE', 'BUT', 'NOT', 'YOU', 'ALL', 'CAN', 'HER', 'WAS', 'ONE', 'OUR',
            'HAD', 'HIS', 'HAS', 'SHE', 'TWO', 'WAY', 'WHO', 'BOY', 'DID', 'ITS', 'LET', 'PUT', 'END',
            'WHY', 'TRY', 'GOD', 'SIX', 'DOG', 'EAT', 'AGO', 'SIT', 'FUN', 'BAD', 'YES', 'YET', 'ARM',
            'FAR', 'OFF', 'BAG', 'BED', 'BOX', 'CAR', 'CUP', 'EGG', 'EYE', 'HAT', 'JOB', 'LEG', 'MAN',
            'PEN', 'SUN', 'TOP', 'WIN', 'AIR', 'BIG', 'BOY', 'BUS', 'CAT', 'COW', 'CRY', 'CUT', 'DAD',
            'DAY', 'DIG', 'EAR', 'FLY', 'GUN', 'HIT', 'HOT', 'HUG', 'ICE', 'KEY', 'LAY', 'LOG', 'MOM',
            'MUD', 'NET', 'NUT', 'OIL', 'PIG', 'POT', 'RUN', 'SAD', 'SEA', 'SKY', 'TOY', 'VAN', 'WAR',
            'WET', 'ZOO',
        ]
        
        # Create frequency map
        self.word_frequencies = {}
        for i, word in enumerate(self.target_words):
            freq = 1.0 - (i * 0.01)
            self.word_frequencies[word] = max(0.1, freq)
        
        self._finalize_setup()
    
    def _finalize_setup(self):
        """Finalize dictionary setup"""
        self.word_set = set(self.target_words)
        print(f"📚 Dictionary finalized: {len(self.target_words)} words")
    
    def levenshtein_distance(self, s1, s2):
        """Calculate Levenshtein distance efficiently"""
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
        """Advanced word correction"""
        if not word:
            return "", 0.0
        
        word = word.upper().strip()
        
        # Direct match
        if word in self.word_set:
            return word, 1.0
        
        # Find best matches
        candidates = []
        
        for candidate in self.target_words:
            distance = self.levenshtein_distance(word, candidate)
            max_len = max(len(word), len(candidate))
            similarity = 1.0 - (distance / max_len) if max_len > 0 else 0.0
            
            # Frequency bonus
            frequency_bonus = self.word_frequencies.get(candidate, 0.5) * 0.1
            
            # Length penalty
            length_penalty = abs(len(word) - len(candidate)) * 0.03
            
            final_score = similarity + frequency_bonus - length_penalty
            final_score = max(0.0, min(1.0, final_score))
            
            candidates.append((candidate, final_score))
        
        # Sort by score
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        if candidates:
            best_word, best_score = candidates[0]
            
            # Try TextBlob if available and score is low
            if TEXTBLOB_AVAILABLE and best_score < 0.6:
                try:
                    blob = TextBlob(word.lower())
                    corrected = str(blob.correct()).upper()
                    if corrected in self.word_set:
                        tb_distance = self.levenshtein_distance(word, corrected)
                        tb_score = max(0.7, 1.0 - (tb_distance / max(len(word), len(corrected))))
                        if tb_score > best_score:
                            return corrected, tb_score
                except Exception as e:
                    print(f"⚠️ TextBlob correction error: {e}")
            
            return best_word, best_score
        
        return word, 0.0
    
    def get_suggestions(self, partial_word, max_suggestions=5):
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

class UltimateVoiceFeedback:
    """Ultimate voice feedback system"""
    
    def __init__(self):
        self.tts_available = TTS_AVAILABLE
        self.engine = None
        self.speaking = False
        self.recent_words = deque(maxlen=10)
        self.last_speech_time = 0
        self.min_speech_interval = 0.8
        
        if TTS_AVAILABLE:
            try:
                self.engine = pyttsx3.init()
                self.configure_engine()
                print("🔊 Ultimate voice feedback initialized")
            except Exception as e:
                print(f"⚠️  TTS initialization failed: {e}")
                self.tts_available = False
    
    def configure_engine(self):
        """Configure TTS engine with optimal settings"""
        if not self.engine:
            return
        
        try:
            # Set optimal speech rate and volume
            self.engine.setProperty('rate', 200)
            self.engine.setProperty('volume', 0.9)
            
            # Try to set a pleasant voice
            voices = self.engine.getProperty('voices')
            if voices:
                for voice in voices:
                    if voice.name and any(keyword in voice.name.lower() 
                                        for keyword in ['female', 'zira', 'hazel', 'samantha']):
                        self.engine.setProperty('voice', voice.id)
                        print(f"✅ Voice set to: {voice.name}")
                        break
        except Exception as e:
            print(f"⚠️  TTS configuration error: {e}")
    
    def should_speak(self, word):
        """Intelligent decision on whether to speak"""
        current_time = time.time()
        
        # Check timing
        if current_time - self.last_speech_time < self.min_speech_interval:
            return False
        
        # Check if recently spoken
        if word in self.recent_words:
            return False
        
        # Check if word is meaningful (not too short)
        if len(word) < 2:
            return False
        
        return True
    
    def speak_word(self, word, priority=False):
        """Speak word with priority handling"""
        if not self.tts_available or not word:
            print(f"🔊 Would speak: {word}")
            return
        
        word = word.strip().upper()
        
        if not priority and not self.should_speak(word):
            return
        
        def speak_thread():
            try:
                self.speaking = True
                print(f"🔊 Speaking: {word}")
                
                if self.engine:
                    # Stop any current speech if priority
                    if priority:
                        try:
                            self.engine.stop()
                        except:
                            pass
                    
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

class UltimateAirWritingSystem:
    """Ultimate air writing recognition system"""
    
    def __init__(self, model_paths=None, dictionary_path=None):
        print("🚀 Initializing Ultimate Air Writing System...")
        
        # Initialize core components
        self.hand_detector = AdvancedHandDetector()
        self.gesture_controller = HandGestureController(self.hand_detector)
        self.letter_recognizer = EnsembleLetterRecognizer(model_paths)
        self.word_corrector = UltimateWordCorrector(dictionary_path)
        self.voice_feedback = UltimateVoiceFeedback()
        
        # Initialize camera with robust error handling
        self.cap = None
        self.initialize_camera()
        
        # Application state
        self.current_word = ""
        self.recognized_words = []
        self.current_path = []
        self.word_suggestions = []
        
        # Timing and thresholds
        self.min_path_length = 8
        self.letter_pause_threshold = 1.2  # seconds
        self.word_pause_threshold = 2.5    # seconds
        self.last_movement_time = time.time()
        self.last_tracking_time = time.time()
        
        # Performance tracking
        self.fps = 0
        self.frame_count = 0
        self.fps_start_time = time.time()
        self.performance_stats = {
            'total_letters': 0,
            'successful_letters': 0,
            'total_words': 0,
            'successful_corrections': 0
        }
        
        # Display settings
        self.show_trail = True
        self.show_debug = False
        self.show_suggestions = True
        
        # Trail visualization
        self.trail_points = deque(maxlen=200)
        self.trail_colors = [(255, 0, 0), (0, 255, 255)]  # Blue to yellow gradient
        
        # Output logging
        self.output_file = "ultimate_airwriting_log.txt"
        self.session_start_time = datetime.now()
        
        print("✅ Ultimate Air Writing System initialized successfully!")
        self.print_instructions()
    
    def initialize_camera(self):
        """Initialize camera with robust error handling"""
        camera_attempts = 3
        
        for attempt in range(camera_attempts):
            try:
                print(f"📹 Initializing camera (attempt {attempt + 1}/{camera_attempts})")
                self.cap = cv2.VideoCapture(0)
                
                if not self.cap.isOpened():
                    raise Exception("Camera not accessible")
                
                # Test camera
                ret, test_frame = self.cap.read()
                if not ret or test_frame is None:
                    raise Exception("Cannot read from camera")
                
                # Set optimal camera properties
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                self.cap.set(cv2.CAP_PROP_FPS, 30)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                
                print("✅ Camera initialized successfully")
                return
                
            except Exception as e:
                print(f"⚠️ Camera initialization attempt {attempt + 1} failed: {e}")
                if self.cap:
                    self.cap.release()
                    self.cap = None
                
                if attempt < camera_attempts - 1:
                    time.sleep(1)
                else:
                    raise Exception("Could not initialize camera after multiple attempts")
    
    def print_instructions(self):
        """Print comprehensive usage instructions"""
        print("\n" + "="*80)
        print("🖐️  ULTIMATE AIR WRITING RECOGNITION SYSTEM")
        print("="*80)
        print("📋 Hand Gesture Controls:")
        print("   ✍️  INDEX FINGER UP (others curled) = Writing mode - tracking active")
        print("   ✋  OPEN HAND (all fingers extended) = Pause tracking")
        print("   ✊  CLOSED HAND (fist) = Pause tracking")
        print("   ✌️  PEACE SIGN (index + middle) = Clear current word")
        print("   👌  OK SIGN (thumb + index circle) = Complete current word")
        print("\n📝 Writing Instructions:")
        print("   • Write letters clearly in the air")
        print("   • System auto-detects letter completion (1.2s pause)")
        print("   • System auto-detects word completion (2.5s pause)")
        print("   • Words are auto-corrected and spoken")
        print("\n⌨️  Keyboard Controls:")
        print("   SPACE    - Force complete current letter")
        print("   ENTER    - Force complete current word")
        print("   C        - Clear current word")
        print("   R        - Reset everything")
        print("   T        - Toggle trail display")
        print("   S        - Toggle word suggestions")
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
        """Process letter completion with advanced logic"""
        if len(self.current_path) < self.min_path_length:
            return
        
        try:
            # Recognize letter using ensemble
            letter, confidence = self.letter_recognizer.recognize_letter(self.current_path)
            
            # Adaptive confidence threshold
            base_threshold = 0.4
            if len(self.current_word) > 0:
                base_threshold = 0.35  # Lower threshold for subsequent letters
            
            # Performance-based adjustment
            if self.performance_stats['total_letters'] > 0:
                success_rate = self.performance_stats['successful_letters'] / self.performance_stats['total_letters']
                if success_rate < 0.6:
                    base_threshold += 0.1  # Raise threshold if accuracy is poor
            
            self.performance_stats['total_letters'] += 1
            
            if letter and confidence > base_threshold:
                self.current_word += letter
                self.performance_stats['successful_letters'] += 1
                
                print(f"✅ Letter: {letter} (confidence: {confidence:.3f})")
                
                # Update suggestions
                self.word_suggestions = self.word_corrector.get_suggestions(self.current_word)
                
                # Show auto-correction preview for longer words
                if len(self.current_word) >= 2:
                    corrected_word, correction_confidence = self.word_corrector.correct_word(self.current_word)
                    if corrected_word != self.current_word and correction_confidence > 0.7:
                        print(f"💡 Auto-correction preview: {self.current_word} → {corrected_word}")
                        if corrected_word not in self.word_suggestions:
                            self.word_suggestions.insert(0, corrected_word)
            else:
                print(f"❌ Letter rejected: {letter} (confidence: {confidence:.3f}, needed: {base_threshold:.3f})")
            
        except Exception as e:
            print(f"⚠️ Letter completion error: {e}")
        finally:
            # Clear path and trail
            self.current_path.clear()
            self.trail_points.clear()
    
    def process_word_completion(self):
        """Process word completion with advanced correction"""
        if not self.current_word:
            return
        
        try:
            # Auto-correct word
            corrected_word, correction_confidence = self.word_corrector.correct_word(self.current_word)
            
            self.performance_stats['total_words'] += 1
            
            if corrected_word != self.current_word:
                print(f"🔄 Auto-correcting: {self.current_word} → {corrected_word} (confidence: {correction_confidence:.3f})")
                self.performance_stats['successful_corrections'] += 1
            else:
                print(f"✅ Word recognized: {corrected_word}")
            
            # Add to recognized words
            self.recognized_words.append(corrected_word)
            
            # Log word
            self.log_word(corrected_word)
            
            # Speak word with priority
            self.voice_feedback.speak_word(corrected_word, priority=True)
            
            # Reset state
            self.reset_word_state()
            
        except Exception as e:
            print(f"⚠️ Word completion error: {e}")
    
    def reset_word_state(self):
        """Reset word-related state"""
        self.current_word = ""
        self.word_suggestions = []
        self.current_path.clear()
        self.trail_points.clear()
    
    def log_word(self, word):
        """Log word to file with timestamp"""
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_entry = f"{timestamp}: {word}\n"
            
            with open(self.output_file, "a", encoding="utf-8") as f:
                f.write(log_entry)
            
            print(f"📝 Logged: {word}")
        except Exception as e:
            print(f"⚠️ Logging error: {e}")
    
    def draw_trail(self, frame):
        """Draw enhanced trail with gradient effects"""
        if len(self.trail_points) < 2:
            return
        
        trail_list = list(self.trail_points)
        
        for i in range(1, len(trail_list)):
            # Calculate progress for gradient
            progress = i / len(trail_list)
            
            # Interpolate color
            color1, color2 = self.trail_colors
            color = tuple(int(c1 + (c2 - c1) * progress) for c1, c2 in zip(color1, color2))
            
            # Variable thickness
            thickness = max(2, int(2 + progress * 4))
            
            cv2.line(frame, trail_list[i-1], trail_list[i], color, thickness)
    
    def draw_ui(self, frame):
        """Draw comprehensive user interface"""
        h, w = frame.shape[:2]
        
        # Create semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (w-10, 280), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        
        # Current word display
        word_display = self.current_word if self.current_word else "[Ready to write...]"
        word_color = (0, 255, 255) if self.current_word else (128, 128, 128)
        cv2.putText(frame, f"Current Word: {word_display}", (20, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, word_color, 3)
        
        # Word suggestions
        if self.show_suggestions and self.word_suggestions:
            suggestions_text = f"Suggestions: {' | '.join(self.word_suggestions[:4])}"
            cv2.putText(frame, suggestions_text, (20, 85), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Recent words
        if self.recognized_words:
            recent_words = " → ".join(self.recognized_words[-4:])
            cv2.putText(frame, f"Recent: {recent_words}", (20, 115), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Performance statistics
        if self.show_debug and self.performance_stats['total_letters'] > 0:
            letter_accuracy = self.performance_stats['successful_letters'] / self.performance_stats['total_letters']
            stats_text = f"Letter Accuracy: {letter_accuracy:.1%} ({self.performance_stats['successful_letters']}/{self.performance_stats['total_letters']})"
            cv2.putText(frame, stats_text, (20, 145), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            if self.performance_stats['total_words'] > 0:
                word_correction_rate = self.performance_stats['successful_corrections'] / self.performance_stats['total_words']
                word_stats_text = f"Word Corrections: {word_correction_rate:.1%} ({self.performance_stats['successful_corrections']}/{self.performance_stats['total_words']})"
                cv2.putText(frame, word_stats_text, (20, 165), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # System status
        status_y = 195
        
        # FPS and performance
        cv2.putText(frame, f"FPS: {self.fps}", (20, status_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Processing time
        avg_proc_time = self.letter_recognizer.get_average_processing_time()
        cv2.putText(frame, f"Processing: {avg_proc_time:.1f}ms", (120, status_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Path length
        cv2.putText(frame, f"Path: {len(self.current_path)}", (280, status_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Voice status
        voice_status = "🔊 Speaking" if self.voice_feedback.is_speaking() else "🔇 Ready"
        cv2.putText(frame, voice_status, (360, status_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255) if self.voice_feedback.is_speaking() else (255, 255, 255), 1)
        
        # Model information
        if self.show_debug:
            model_info = self.letter_recognizer.get_model_info()
            model_text = f"Models: {model_info['total_models']} (CNN: {model_info['cnn_models']}, ML: {model_info['sklearn_models']})"
            cv2.putText(frame, model_text, (20, 225), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Instructions
        instructions = [
            "✍️ INDEX finger = Write | ✋ OPEN hand = Pause | ✊ CLOSED hand = Pause",
            "⌨️ SPACE: Letter | ENTER: Word | C: Clear | T: Trail | S: Suggestions | ESC: Exit"
        ]
        
        for i, instruction in enumerate(instructions):
            cv2.putText(frame, instruction, (20, h - 50 + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    
    def handle_keyboard_input(self, key):
        """Handle keyboard input with comprehensive controls"""
        if key == ord(' '):  # SPACE - force letter completion
            self.process_letter_completion()
        elif key == 13:  # ENTER - force word completion
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
        elif key == ord('s') or key == ord('S'):  # Toggle suggestions
            self.show_suggestions = not self.show_suggestions
            print(f"💡 Word suggestions: {'ON' if self.show_suggestions else 'OFF'}")
        elif key == ord('d') or key == ord('D'):  # Toggle debug
            self.show_debug = not self.show_debug
            print(f"🐛 Debug information: {'ON' if self.show_debug else 'OFF'}")
        elif key == ord('v') or key == ord('V'):  # Test voice
            print("🧪 Testing voice feedback...")
            self.voice_feedback.speak_word("HELLO", priority=True)
        elif key == 27:  # ESC - exit
            return False
        
        return True
    
    def run(self):
        """Main application loop with robust error handling"""
        print("🚀 Starting Ultimate Air Writing System...")
        
        frame_counter = 0
        last_successful_frame = time.time()
        
        try:
            while True:
                # Read frame with error handling
                ret, frame = self.cap.read()
                if not ret:
                    print("❌ Failed to read frame")
                    break
                
                frame_counter += 1
                last_successful_frame = time.time()
                
                # Flip frame for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Process hand detection
                try:
                    processed_frame, hand_info = self.hand_detector.process_frame(frame)
                    frame = processed_frame
                except Exception as e:
                    print(f"⚠️ Hand detection error: {e}")
                    hand_info = {'present': False, 'state': HandState.UNKNOWN, 'fingertip': None}
                
                # Update FPS
                self.update_fps()
                
                # Handle writing logic
                current_time = time.time()
                
                if hand_info['present'] and hand_info['state'] == HandState.WRITING:
                    # Active writing mode
                    if hand_info['fingertip']:
                        self.current_path.append(hand_info['fingertip'])
                        self.trail_points.append(hand_info['fingertip'])
                        self.last_movement_time = current_time
                        self.last_tracking_time = current_time
                else:
                    # Handle pauses and gesture commands
                    time_since_movement = current_time - self.last_movement_time
                    
                    # Process gesture commands
                    try:
                        command = self.gesture_controller.process_gestures(hand_info)
                        if command == "CLEAR_WORD":
                            self.reset_word_state()
                        elif command == "COMPLETE_WORD" and self.current_word:
                            self.process_word_completion()
                    except Exception as e:
                        print(f"⚠️ Gesture processing error: {e}")
                    
                    # Auto letter completion
                    if (len(self.current_path) >= self.min_path_length and 
                        time_since_movement > self.letter_pause_threshold):
                        self.process_letter_completion()
                    
                    # Auto word completion
                    if (self.current_word and 
                        time_since_movement > self.word_pause_threshold):
                        print(f"⏰ Auto-completing word after {time_since_movement:.1f}s pause")
                        self.process_word_completion()
                
                # Draw trail
                if self.show_trail:
                    try:
                        self.draw_trail(frame)
                    except Exception as e:
                        print(f"⚠️ Trail drawing error: {e}")
                
                # Draw UI
                try:
                    self.draw_ui(frame)
                except Exception as e:
                    print(f"⚠️ UI drawing error: {e}")
                
                # Show frame
                try:
                    cv2.imshow('Ultimate Air Writing System', frame)
                except Exception as e:
                    print(f"⚠️ Frame display error: {e}")
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if not self.handle_keyboard_input(key):
                    break
                
                # System stability check
                if frame_counter % 300 == 0:  # Every 10 seconds at 30 FPS
                    print(f"📊 System stable - Frame {frame_counter}, FPS: {self.fps}")
        
        except KeyboardInterrupt:
            print("\n👋 System interrupted by user")
        except Exception as e:
            print(f"❌ Critical system error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Comprehensive system cleanup"""
        print("🧹 Cleaning up Ultimate Air Writing System...")
        
        try:
            # Stop voice feedback
            if hasattr(self, 'voice_feedback') and self.voice_feedback:
                if hasattr(self.voice_feedback, 'engine') and self.voice_feedback.engine:
                    self.voice_feedback.engine.stop()
                print("✅ Voice feedback stopped")
        except Exception as e:
            print(f"⚠️ Voice cleanup error: {e}")
        
        try:
            # Release camera
            if hasattr(self, 'cap') and self.cap:
                self.cap.release()
                print("✅ Camera released")
        except Exception as e:
            print(f"⚠️ Camera cleanup error: {e}")
        
        try:
            # Close all windows
            cv2.destroyAllWindows()
            print("✅ Windows closed")
        except Exception as e:
            print(f"⚠️ Window cleanup error: {e}")
        
        # Print session summary
        try:
            session_duration = datetime.now() - self.session_start_time
            print(f"\n📊 Session Summary:")
            print(f"   Duration: {session_duration}")
            print(f"   Words recognized: {len(self.recognized_words)}")
            print(f"   Letters processed: {self.performance_stats['total_letters']}")
            print(f"   Letter accuracy: {self.performance_stats['successful_letters']}/{self.performance_stats['total_letters']}")
            if self.recognized_words:
                print(f"   Final words: {' '.join(self.recognized_words[-5:])}")
        except Exception as e:
            print(f"⚠️ Summary generation error: {e}")
        
        print("✅ Cleanup complete")

def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(description="Ultimate Air Writing Recognition System")
    parser.add_argument('--models', nargs='+', help='Paths to model files')
    parser.add_argument('--dictionary', type=str, help='Path to dictionary file')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--list-models', action='store_true', help='List available models')
    
    args = parser.parse_args()
    
    if args.list_models:
        print("\n📁 Available Models:")
        model_dir = Path("models")
        if model_dir.exists():
            for model_file in model_dir.glob("*.h5"):
                print(f"   ✅ {model_file}")
            for model_file in model_dir.glob("*.joblib"):
                print(f"   ✅ {model_file}")
        else:
            print("   ❌ No models directory found")
        return
    
    print("🖐️  ULTIMATE AIR WRITING RECOGNITION SYSTEM")
    print("=" * 70)
    print("🎯 Advanced hand gesture recognition with open/closed detection")
    print("🧠 Ensemble letter recognition with multiple models")
    print("📚 Smart word auto-correction and completion")
    print("🔊 Intelligent voice feedback")
    print("📊 Real-time performance monitoring")
    print("=" * 70)
    
    if args.models:
        print(f"📁 Using models: {args.models}")
    if args.dictionary:
        print(f"📚 Using dictionary: {args.dictionary}")
    if args.debug:
        print("🐛 Debug mode enabled")
    
    try:
        system = UltimateAirWritingSystem(
            model_paths=args.models,
            dictionary_path=args.dictionary
        )
        system.run()
        
    except KeyboardInterrupt:
        print("\n👋 System interrupted by user")
    except Exception as e:
        print(f"❌ System initialization error: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
    finally:
        print("🏁 Ultimate Air Writing System finished!")

if __name__ == "__main__":
    main()