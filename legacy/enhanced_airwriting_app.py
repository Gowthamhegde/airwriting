#!/usr/bin/env python3
"""
Enhanced Real-time Air Writing Recognition System
Advanced word recognition with improved accuracy and user experience
"""

import cv2
import numpy as np
import time
import threading
import json
from collections import deque
from keras.models import load_model
from utils.hand_tracker import HandTracker
from utils.preprocessing import (
    draw_path_on_blank, enhance_letter_image, 
    extract_letter_region, calculate_path_features
)
from word_recognition import correct_word_enhanced, get_word_suggestions, get_word_confidence
from text_to_speech import speak_word

class EnhancedAirWritingSystem:
    def __init__(self):
        # Load the trained model
        try:
            self.model = load_model("models/letter_recognition.h5")
            print("✓ Letter recognition model loaded successfully")
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            return
        
        # Initialize enhanced hand tracker
        self.tracker = HandTracker(max_hands=1, trail_length=150, alpha=0.2)
        
        # Initialize camera with higher resolution
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Writing state variables
        self.current_path = []
        self.current_word = ""
        self.recognized_words = []
        self.word_history = deque(maxlen=50)
        self.last_movement_time = time.time()
        
        # Enhanced timing parameters
        self.LETTER_PAUSE_THRESHOLD = 20  # frames with low velocity to end letter
        self.WORD_PAUSE_THRESHOLD = 120  # frames to end word
        self.VELOCITY_THRESHOLD = 2.5    # minimum velocity to consider movement
        self.WORD_COMPLETION_PAUSE = 3.5 # seconds to finalize word
        self.MIN_PATH_LENGTH = 10        # minimum points for letter recognition
        
        # Recognition parameters
        self.confidence_threshold = 0.3  # Lower threshold for better recognition
        self.min_word_confidence = 0.2   # Lower for short words
        
        # Frame counters
        self.letter_pause_frames = 0
        self.word_pause_frames = 0
        
        # UI state
        self.show_trail = True
        self.show_debug = True
        self.show_suggestions = True
        self.auto_correct = True
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        
        # Statistics
        self.stats = {
            'letters_recognized': 0,
            'words_completed': 0,
            'session_start': time.time(),
            'total_writing_time': 0
        }
        
        # Letter prediction history for context
        self.letter_predictions = deque(maxlen=10)
        
        print("✓ Enhanced Air Writing System initialized")
        self.print_instructions()
    
    def print_instructions(self):
        """Print usage instructions"""
        print("\n" + "="*60)
        print("ENHANCED AIR WRITING SYSTEM - INSTRUCTIONS")
        print("="*60)
        print("✋ Hold up your index finger to start writing")
        print("✍️  Write letters in the air with your index finger")
        print("⏸️  Pause briefly between letters")
        print("⏹️  Pause longer to complete words")
        print("\nKEYBOARD CONTROLS:")
        print("  SPACE  - End current letter manually")
        print("  S      - Speak current word")
        print("  C      - Clear current word")
        print("  T      - Toggle trail visibility")
        print("  D      - Toggle debug information")
        print("  A      - Toggle auto-correction")
        print("  H      - Show/hide word suggestions")
        print("  R      - Reset statistics")
        print("  ESC    - Exit application")
        print("="*60 + "\n")
    
    def process_letter_advanced(self, path):
        """Advanced letter processing with multiple techniques"""
        if len(path) < self.MIN_PATH_LENGTH:
            return None, 0.0, {}
        
        try:
            # Calculate path features for additional context
            features = calculate_path_features(path)
            
            # Create image from path
            img = draw_path_on_blank(path, img_size=256)
            
            # Apply image enhancement
            enhanced_img = enhance_letter_image(img)
            
            # Extract letter region
            letter_img = extract_letter_region(enhanced_img)
            
            # Resize and normalize for model
            img_resized = cv2.resize(letter_img, (28, 28))
            img_normalized = img_resized.reshape(1, 28, 28, 1) / 255.0
            
            # Predict letter
            prediction = self.model.predict(img_normalized, verbose=0)
            confidence = np.max(prediction)
            letter_idx = np.argmax(prediction)
            letter = chr(letter_idx + ord("A"))
            
            # Get top 3 predictions for context
            top_indices = np.argsort(prediction[0])[-3:][::-1]
            top_predictions = [(chr(idx + ord("A")), prediction[0][idx]) 
                             for idx in top_indices]
            
            # Store prediction history
            self.letter_predictions.append({
                'letter': letter,
                'confidence': confidence,
                'top_predictions': top_predictions,
                'features': features,
                'timestamp': time.time()
            })
            
            return letter, confidence, features
            
        except Exception as e:
            print(f"Error in letter processing: {e}")
            return None, 0.0, {}
    
    def get_contextual_correction(self, letter, confidence, features):
        """Apply contextual correction based on writing patterns"""
        # If confidence is very high, trust the prediction
        if confidence > 0.8:
            return letter
        
        # Check recent predictions for patterns
        if len(self.letter_predictions) >= 2:
            recent = self.letter_predictions[-2]
            
            # Common letter confusions based on writing patterns
            confusion_map = {
                ('O', 'D'): lambda f: 'O' if f.get('aspect_ratio', 1) > 1.2 else 'D',
                ('I', 'L'): lambda f: 'I' if f.get('height', 0) > f.get('width', 1) * 2 else 'L',
                ('C', 'G'): lambda f: 'C' if f.get('direction_changes', 0) < 2 else 'G',
                ('U', 'V'): lambda f: 'U' if f.get('direction_changes', 0) < 3 else 'V',
            }
            
            # Apply contextual rules
            for (l1, l2), rule in confusion_map.items():
                if letter in [l1, l2]:
                    corrected = rule(features)
                    if corrected != letter:
                        print(f"Contextual correction: {letter} -> {corrected}")
                        return corrected
        
        return letter
    
    def update_fps(self):
        """Update FPS counter"""
        self.fps_counter += 1
        if time.time() - self.fps_start_time >= 1.0:
            self.current_fps = self.fps_counter
            self.fps_counter = 0
            self.fps_start_time = time.time()
    
    def draw_enhanced_ui(self, frame, fingertip, velocity):
        """Draw enhanced user interface"""
        h, w = frame.shape[:2]
        
        # Create semi-transparent overlay
        overlay = frame.copy()
        
        # Main word display area
        cv2.rectangle(overlay, (10, 10), (w-10, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        
        # Current word with larger font
        word_text = f"Word: {self.current_word}"
        cv2.putText(frame, word_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
        
        # Word suggestions
        if self.show_suggestions and self.current_word:
            suggestions = get_word_suggestions(self.current_word)
            if suggestions:
                suggestion_text = f"Suggestions: {' | '.join(suggestions[:3])}"
                cv2.putText(frame, suggestion_text, (20, 85), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
        
        # Recent words
        if self.recognized_words:
            recent_words = " | ".join(self.recognized_words[-4:])
            cv2.putText(frame, f"Recent: {recent_words}", (20, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Debug information
        if self.show_debug:
            debug_y = h - 180
            debug_info = [
                f"FPS: {self.current_fps}",
                f"Velocity: {velocity:.1f}",
                f"Path Points: {len(self.current_path)}",
                f"Letter Pause: {self.letter_pause_frames}",
                f"Word Pause: {self.word_pause_frames}",
                f"Writing Mode: {'Yes' if self.tracker.is_writing_active() else 'No'}",
                f"Gesture Confidence: {self.tracker.gesture_confidence:.2f}"
            ]
            
            for i, info in enumerate(debug_info):
                cv2.putText(frame, info, (20, debug_y + i * 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Statistics
        session_time = time.time() - self.stats['session_start']
        stats_text = [
            f"Session: {int(session_time//60)}:{int(session_time%60):02d}",
            f"Letters: {self.stats['letters_recognized']}",
            f"Words: {self.stats['words_completed']}"
        ]
        
        for i, stat in enumerate(stats_text):
            cv2.putText(frame, stat, (w - 200, 30 + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        
        # Instructions panel
        instructions = [
            "SPACE: End letter | S: Speak | C: Clear",
            "T: Toggle trail | D: Debug | A: Auto-correct",
            "H: Suggestions | R: Reset stats | ESC: Exit"
        ]
        
        for i, instruction in enumerate(instructions):
            cv2.putText(frame, instruction, (w - 400, h - 60 + i * 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
        
        # Status indicators
        status_y = h - 120
        if self.auto_correct:
            cv2.putText(frame, "AUTO-CORRECT: ON", (w - 200, status_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        if self.show_suggestions:
            cv2.putText(frame, "SUGGESTIONS: ON", (w - 200, status_y + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    def save_session_data(self):
        """Save session statistics and word history"""
        session_data = {
            'timestamp': time.time(),
            'stats': self.stats,
            'word_history': list(self.word_history),
            'recognized_words': self.recognized_words
        }
        
        try:
            with open('session_data.json', 'w') as f:
                json.dump(session_data, f, indent=2)
            print("Session data saved to session_data.json")
        except Exception as e:
            print(f"Error saving session data: {e}")
    
    def run(self):
        """Enhanced main application loop"""
        print("Starting Enhanced Air Writing System...")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to read from camera")
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Get fingertip position and velocity from enhanced tracker
            fingertip, annotated_frame, velocity = self.tracker.get_fingertip(frame)
            
            # Update FPS
            self.update_fps()
            
            # Process fingertip movement only in writing mode
            if fingertip and self.tracker.is_writing_active():
                self.current_path.append(fingertip)
                self.last_movement_time = time.time()
            
            # Automatic letter segmentation based on velocity
            if velocity < self.VELOCITY_THRESHOLD:
                self.letter_pause_frames += 1
            else:
                self.letter_pause_frames = 0
            
            # Process letter when movement stops
            if (self.letter_pause_frames > self.LETTER_PAUSE_THRESHOLD and 
                len(self.current_path) >= self.MIN_PATH_LENGTH):
                
                letter, confidence, features = self.process_letter_advanced(self.current_path)
                
                if letter and confidence > self.confidence_threshold:
                    # Apply contextual correction
                    corrected_letter = self.get_contextual_correction(letter, confidence, features)
                    
                    self.current_word += corrected_letter
                    self.stats['letters_recognized'] += 1
                    
                    print(f"Letter: {corrected_letter} (confidence: {confidence:.2f})")
                    
                    if corrected_letter != letter:
                        print(f"  Contextually corrected from: {letter}")
                else:
                    print(f"Low confidence letter rejected: {letter} ({confidence:.2f})")
                
                self.current_path.clear()
                self.letter_pause_frames = 0
            
            # Automatic word completion based on pause
            time_since_movement = time.time() - self.last_movement_time
            if time_since_movement > self.WORD_COMPLETION_PAUSE:
                self.word_pause_frames += 1
            else:
                self.word_pause_frames = 0
            
            # Finalize word after long pause
            if (self.word_pause_frames > self.WORD_PAUSE_THRESHOLD and 
                self.current_word and len(self.current_word) > 0):
                
                self.finalize_word()
            
            # Draw enhanced UI
            self.draw_enhanced_ui(annotated_frame, fingertip, velocity)
            
            # Show frame
            cv2.imshow("Enhanced Air Writing Recognition System", annotated_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' '):  # Manual letter end
                if len(self.current_path) >= self.MIN_PATH_LENGTH:
                    letter, confidence, features = self.process_letter_advanced(self.current_path)
                    if letter and confidence > self.confidence_threshold:
                        corrected_letter = self.get_contextual_correction(letter, confidence, features)
                        self.current_word += corrected_letter
                        self.stats['letters_recognized'] += 1
                        print(f"Manual letter: {corrected_letter} (confidence: {confidence:.2f})")
                self.current_path.clear()
                self.letter_pause_frames = 0
            
            elif key == ord('s'):  # Speak current word
                if self.current_word:
                    self.finalize_word(speak=True)
            
            elif key == ord('c'):  # Clear current word
                self.current_word = ""
                self.current_path.clear()
                print("Word cleared")
            
            elif key == ord('t'):  # Toggle trail visibility
                self.show_trail = not self.show_trail
                print(f"Trail visibility: {self.show_trail}")
            
            elif key == ord('d'):  # Toggle debug info
                self.show_debug = not self.show_debug
                print(f"Debug info: {self.show_debug}")
            
            elif key == ord('a'):  # Toggle auto-correction
                self.auto_correct = not self.auto_correct
                print(f"Auto-correction: {self.auto_correct}")
            
            elif key == ord('h'):  # Toggle suggestions
                self.show_suggestions = not self.show_suggestions
                print(f"Word suggestions: {self.show_suggestions}")
            
            elif key == ord('r'):  # Reset statistics
                self.reset_stats()
            
            elif key == 27:  # ESC to exit
                break
        
        # Save session data before exit
        self.save_session_data()
        
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()
        print("Enhanced Air Writing System stopped")
    
    def finalize_word(self, speak=False):
        """Finalize the current word with enhanced processing"""
        if not self.current_word:
            return
        
        original_word = self.current_word
        
        # Apply auto-correction if enabled
        if self.auto_correct:
            corrected_word = correct_word_enhanced(self.current_word)
            word_confidence = get_word_confidence(corrected_word)
            
            if word_confidence > self.min_word_confidence:
                final_word = corrected_word
                if corrected_word != original_word:
                    print(f"Auto-corrected: {original_word} -> {corrected_word}")
            else:
                final_word = original_word
                print(f"Auto-correction skipped (low confidence: {word_confidence:.2f})")
        else:
            final_word = original_word
        
        # Add to history
        self.recognized_words.append(final_word)
        self.word_history.append({
            'original': original_word,
            'corrected': final_word,
            'timestamp': time.time(),
            'letter_count': len(original_word)
        })
        
        self.stats['words_completed'] += 1
        
        print(f"Word completed: {final_word}")
        
        # Speak word if requested
        if speak:
            threading.Thread(target=speak_word, args=(final_word,), daemon=True).start()
        
        # Reset for next word
        self.current_word = ""
        self.word_pause_frames = 0
        self.current_path.clear()
    
    def reset_stats(self):
        """Reset session statistics"""
        self.stats = {
            'letters_recognized': 0,
            'words_completed': 0,
            'session_start': time.time(),
            'total_writing_time': 0
        }
        self.word_history.clear()
        self.recognized_words.clear()
        print("Statistics reset")

def main():
    """Main entry point"""
    try:
        system = EnhancedAirWritingSystem()
        system.run()
    except KeyboardInterrupt:
        print("\nSystem interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()