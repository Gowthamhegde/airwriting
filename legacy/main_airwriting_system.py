#!/usr/bin/env python3
"""
Real-time Air Writing Recognition System
Recognizes words using index finger gestures with enhanced accuracy and features
"""

import cv2
import numpy as np
import time
import threading
from collections import deque
from keras.models import load_model
from utils.hand_tracker import HandTracker
from utils.preprocessing import draw_path_on_blank
from autocorrector import correct_word
from text_to_speech import speak_word

class AirWritingSystem:
    def __init__(self):
        # Load the trained model
        try:
            self.model = load_model("models/letter_recognition.h5")
            print("✓ Model loaded successfully")
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            return
        
        # Initialize hand tracker
        self.tracker = HandTracker(max_hands=1, trail_length=100)
        
        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Writing state variables
        self.path = []
        self.current_word = ""
        self.recognized_words = []
        self.last_movement_time = time.time()
        
        # Timing and threshold parameters
        self.LETTER_PAUSE_THRESHOLD = 15  # frames with low velocity to end letter
        self.WORD_PAUSE_THRESHOLD = 90   # frames to end word
        self.VELOCITY_THRESHOLD = 3.0    # minimum velocity to consider movement
        self.WORD_COMPLETION_PAUSE = 3.0 # seconds to finalize word
        self.MIN_PATH_LENGTH = 8         # minimum points for letter recognition
        
        # Frame counters
        self.letter_pause_frames = 0
        self.word_pause_frames = 0
        
        # UI state
        self.show_trail = True
        self.show_debug = True
        self.confidence_threshold = 0.6
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        
        print("✓ Air Writing System initialized")
    
    def process_letter(self, path):
        """Process a drawn path and predict the letter"""
        if len(path) < self.MIN_PATH_LENGTH:
            return None, 0.0
        
        try:
            # Create image from path
            img = draw_path_on_blank(path, img_size=256)
            
            # Resize and normalize for model
            img_resized = cv2.resize(img, (28, 28))
            img_normalized = img_resized.reshape(1, 28, 28, 1) / 255.0
            
            # Predict letter
            prediction = self.model.predict(img_normalized, verbose=0)
            confidence = np.max(prediction)
            letter_idx = np.argmax(prediction)
            letter = chr(letter_idx + ord("A"))
            
            return letter, confidence
        except Exception as e:
            print(f"Error in letter processing: {e}")
            return None, 0.0
    
    def update_fps(self):
        """Update FPS counter"""
        self.fps_counter += 1
        if time.time() - self.fps_start_time >= 1.0:
            self.current_fps = self.fps_counter
            self.fps_counter = 0
            self.fps_start_time = time.time()
    
    def draw_ui(self, frame, fingertip, velocity):
        """Draw user interface elements on frame"""
        h, w = frame.shape[:2]
        
        # Create semi-transparent overlay for text
        overlay = frame.copy()
        
        # Draw current word area
        cv2.rectangle(overlay, (10, 10), (w-10, 100), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Display current word
        word_text = f"Word: {self.current_word}"
        cv2.putText(frame, word_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)
        
        # Display recognized words
        if self.recognized_words:
            recent_words = " | ".join(self.recognized_words[-3:])  # Show last 3 words
            cv2.putText(frame, f"Recent: {recent_words}", (20, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        if self.show_debug:
            # Debug information
            debug_y = h - 120
            cv2.putText(frame, f"FPS: {self.current_fps}", (20, debug_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(frame, f"Velocity: {velocity:.1f}", (20, debug_y + 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(frame, f"Path Points: {len(self.path)}", (20, debug_y + 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(frame, f"Letter Pause: {self.letter_pause_frames}", (20, debug_y + 75), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Instructions
        instructions = [
            "SPACE: End letter manually",
            "S: Speak current word", 
            "C: Clear word",
            "T: Toggle trail",
            "D: Toggle debug",
            "ESC: Exit"
        ]
        
        for i, instruction in enumerate(instructions):
            cv2.putText(frame, instruction, (w - 250, 30 + i * 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    
    def run(self):
        """Main application loop"""
        print("Starting Air Writing System...")
        print("Use your index finger to write letters in the air!")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to read from camera")
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Get fingertip position and velocity
            fingertip, annotated_frame, velocity = self.tracker.get_fingertip(frame)
            
            # Update FPS
            self.update_fps()
            
            # Process fingertip movement
            if fingertip:
                self.path.append(fingertip)
                self.last_movement_time = time.time()
            
            # Automatic letter segmentation based on velocity
            if velocity < self.VELOCITY_THRESHOLD:
                self.letter_pause_frames += 1
            else:
                self.letter_pause_frames = 0
            
            # Process letter when movement stops
            if (self.letter_pause_frames > self.LETTER_PAUSE_THRESHOLD and 
                len(self.path) >= self.MIN_PATH_LENGTH):
                
                letter, confidence = self.process_letter(self.path)
                
                if letter and confidence > self.confidence_threshold:
                    self.current_word += letter
                    print(f"Letter recognized: {letter} (confidence: {confidence:.2f})")
                else:
                    print(f"Low confidence letter rejected: {letter} ({confidence:.2f})")
                
                self.path.clear()
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
                
                corrected_word = correct_word(self.current_word)
                print(f"Word completed: {self.current_word} -> {corrected_word}")
                
                self.recognized_words.append(corrected_word)
                
                # Speak word in separate thread to avoid blocking
                threading.Thread(target=speak_word, args=(corrected_word,), daemon=True).start()
                
                self.current_word = ""
                self.word_pause_frames = 0
                self.path.clear()
            
            # Draw UI
            self.draw_ui(annotated_frame, fingertip, velocity)
            
            # Show frame
            cv2.imshow("Air Writing Recognition System", annotated_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' '):  # Manual letter end
                if len(self.path) >= self.MIN_PATH_LENGTH:
                    letter, confidence = self.process_letter(self.path)
                    if letter and confidence > self.confidence_threshold:
                        self.current_word += letter
                        print(f"Manual letter: {letter} (confidence: {confidence:.2f})")
                self.path.clear()
                self.letter_pause_frames = 0
            
            elif key == ord('s'):  # Speak current word
                if self.current_word:
                    corrected_word = correct_word(self.current_word)
                    print(f"Speaking: {self.current_word} -> {corrected_word}")
                    self.recognized_words.append(corrected_word)
                    threading.Thread(target=speak_word, args=(corrected_word,), daemon=True).start()
                    self.current_word = ""
                self.path.clear()
            
            elif key == ord('c'):  # Clear current word
                self.current_word = ""
                self.path.clear()
                print("Word cleared")
            
            elif key == ord('t'):  # Toggle trail visibility
                self.show_trail = not self.show_trail
                print(f"Trail visibility: {self.show_trail}")
            
            elif key == ord('d'):  # Toggle debug info
                self.show_debug = not self.show_debug
                print(f"Debug info: {self.show_debug}")
            
            elif key == 27:  # ESC to exit
                break
        
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()
        print("Air Writing System stopped")

def main():
    """Main entry point"""
    try:
        system = AirWritingSystem()
        system.run()
    except KeyboardInterrupt:
        print("\nSystem interrupted by user")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()