#!/usr/bin/env python3
"""
Simple Word Air Writing System
Optimized for recognizing simple words like CAT, DOG, etc.
Uses improved hand tracking and advanced model
"""

import cv2
import numpy as np
import time
import json
import os
from collections import deque
from improved_hand_tracker import ImprovedHandTracker
from utils.preprocessing import draw_path_on_blank, enhance_letter_image
from text_to_speech import speak_word

# Try to import TensorFlow/Keras
try:
    from keras.models import load_model
    MODEL_AVAILABLE = True
except ImportError:
    print("⚠️  TensorFlow/Keras not available - running in demo mode")
    MODEL_AVAILABLE = False

class SimpleWordAirWriting:
    def __init__(self):
        # Initialize improved hand tracker
        self.tracker = ImprovedHandTracker()
        
        # Load model if available
        self.model = None
        if MODEL_AVAILABLE:
            self.load_model()
        
        # Load word dictionary
        self.load_word_dictionary()
        
        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise Exception("Could not open camera")
        
        # Set camera properties for better quality
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Writing state
        self.current_word = ""
        self.recognized_words = []
        self.current_path = []
        self.letter_paths = []
        
        # Timing parameters (optimized for better accuracy)
        self.LETTER_PAUSE_FRAMES = 30    # Longer pause to ensure letter completion
        self.WORD_PAUSE_FRAMES = 90      # Pause between words
        self.MIN_PATH_LENGTH = 15        # Minimum points for a valid letter
        self.CONFIDENCE_THRESHOLD = 0.3   # Lower threshold for simple words
        
        # State tracking
        self.letter_pause_count = 0
        self.word_pause_count = 0
        self.last_movement_time = time.time()
        
        # Performance tracking
        self.fps = 0
        self.frame_count = 0
        self.fps_start_time = time.time()
        
        print("✅ Simple Word Air Writing System initialized")
        if self.model is not None:
            print(f"🧠 Model loaded with {len(self.target_words)} target words")
        else:
            print("⚠️  Running in demo mode (no model)")
    
    def load_model(self):
        """Load the trained model"""
        model_paths = [
            "models/letter_recognition_advanced.h5",
            "models/advanced_letter_recognition.h5",
            "models/letter_recognition_enhanced.h5",
            "models/letter_recognition.h5"
        ]
        
        for model_path in model_paths:
            if os.path.exists(model_path):
                try:
                    self.model = load_model(model_path)
                    print(f"✅ Loaded model: {model_path}")
                    return
                except Exception as e:
                    print(f"⚠️  Failed to load {model_path}: {e}")
                    continue
        
        print("❌ No model found. Please train a model first.")
        print("   Run: python train_advanced_model.py")
    
    def load_word_dictionary(self):
        """Load the word dictionary"""
        self.target_words = [
            'CAT', 'DOG', 'BAT', 'RAT', 'HAT', 'MAT', 'SAT', 'FAT',
            'BIG', 'PIG', 'DIG', 'FIG', 'WIG',
            'SUN', 'RUN', 'FUN', 'GUN', 'BUN',
            'BOX', 'FOX', 'SOX',
            'BED', 'RED', 'LED', 'FED',
            'TOP', 'HOP', 'MOP', 'POP', 'COP',
            'CUP', 'PUP',
            'BAG', 'TAG', 'RAG', 'SAG', 'WAG',
            'BUS', 'YES', 'NET', 'PET', 'SET', 'WET', 'GET', 'LET', 'MET',
            'HOT', 'POT', 'COT', 'DOT', 'GOT', 'LOT', 'NOT', 'ROT',
            'BAD', 'DAD', 'HAD', 'MAD', 'PAD', 'SAD',
            'BEE', 'SEE', 'TEE', 'FEE',
            'EGG', 'LEG', 'BEG', 'PEG',
            'ICE'
        ]
        
        # Try to load from file
        dict_path = "models/word_dictionary.json"
        if os.path.exists(dict_path):
            try:
                with open(dict_path, 'r') as f:
                    data = json.load(f)
                    self.target_words = data.get('target_words', self.target_words)
                print(f"✅ Loaded {len(self.target_words)} target words from dictionary")
            except Exception as e:
                print(f"⚠️  Could not load dictionary: {e}")
        
        # Create word lookup for fast correction
        self.word_set = set(self.target_words)
    
    def recognize_letter(self, path):
        """Recognize a letter from the drawn path"""
        if self.model is None:
            # Demo mode - return random letter
            return chr(ord('A') + np.random.randint(0, 26)), 0.5
        
        if len(path) < self.MIN_PATH_LENGTH:
            return None, 0.0
        
        try:
            # Create image from path
            img = draw_path_on_blank(path, img_size=256)
            
            # Enhance image
            enhanced_img = enhance_letter_image(img)
            
            # Resize for model (check model input size)
            model_input_size = 32 if 'advanced' in str(self.model) else 28
            img_resized = cv2.resize(enhanced_img, (model_input_size, model_input_size))
            img_normalized = img_resized.reshape(1, model_input_size, model_input_size, 1) / 255.0
            
            # Predict
            prediction = self.model.predict(img_normalized, verbose=0)
            confidence = np.max(prediction)
            letter_idx = np.argmax(prediction)
            letter = chr(letter_idx + ord('A'))
            
            return letter, confidence
            
        except Exception as e:
            print(f"Error in letter recognition: {e}")
            return None, 0.0
    
    def correct_word(self, word):
        """Correct word using simple dictionary lookup"""
        if not word:
            return ""
        
        # Direct match
        if word in self.word_set:
            return word
        
        # Find closest match
        best_match = word
        best_score = 0
        
        for target_word in self.target_words:
            if len(target_word) == len(word):
                # Calculate similarity (simple character matching)
                matches = sum(1 for a, b in zip(word, target_word) if a == b)
                score = matches / len(word)
                
                if score > best_score and score > 0.5:  # At least 50% match
                    best_score = score
                    best_match = target_word
        
        return best_match
    
    def update_fps(self):
        """Update FPS counter"""
        self.frame_count += 1
        current_time = time.time()
        
        if current_time - self.fps_start_time >= 1.0:
            self.fps = self.frame_count
            self.frame_count = 0
            self.fps_start_time = current_time
    
    def draw_ui(self, frame):
        """Draw user interface"""
        h, w = frame.shape[:2]
        
        # Main display area
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (w-10, 150), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        
        # Current word (large text)
        word_text = f"Word: {self.current_word}"
        cv2.putText(frame, word_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
        
        # Recognized words
        if self.recognized_words:
            recent_words = " | ".join(self.recognized_words[-3:])
            cv2.putText(frame, f"Recent: {recent_words}", (20, 85), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Word suggestions
        if self.current_word:
            suggestions = [word for word in self.target_words 
                          if word.startswith(self.current_word)][:3]
            if suggestions:
                suggestion_text = f"Suggestions: {' | '.join(suggestions)}"
                cv2.putText(frame, suggestion_text, (20, 115), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
        
        # Instructions
        instructions = [
            "Hold index finger up and write letters in air",
            "Pause between letters, longer pause between words",
            "SPACE: End letter | S: Speak word | C: Clear | ESC: Exit"
        ]
        
        for i, instruction in enumerate(instructions):
            cv2.putText(frame, instruction, (20, h - 80 + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Status info
        status_y = h - 150
        cv2.putText(frame, f"FPS: {self.fps}", (w - 150, status_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f"Letters: {len(self.current_word)}", (w - 150, status_y + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f"Words: {len(self.recognized_words)}", (w - 150, status_y + 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Target words panel
        self.draw_target_words_panel(frame)
    
    def draw_target_words_panel(self, frame):
        """Draw panel showing target words"""
        h, w = frame.shape[:2]
        
        # Panel background
        panel_x = w - 300
        panel_y = 10
        panel_w = 280
        panel_h = 200
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Title
        cv2.putText(frame, "Target Words:", (panel_x + 10, panel_y + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Show some target words
        words_to_show = self.target_words[:15]  # Show first 15 words
        cols = 3
        rows = 5
        
        for i, word in enumerate(words_to_show):
            row = i // cols
            col = i % cols
            
            x = panel_x + 15 + col * 85
            y = panel_y + 50 + row * 25
            
            # Highlight if current word matches
            color = (0, 255, 0) if word == self.current_word else (200, 200, 200)
            cv2.putText(frame, word, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    def process_letter_completion(self):
        """Process when a letter is completed"""
        if len(self.current_path) < self.MIN_PATH_LENGTH:
            return
        
        letter, confidence = self.recognize_letter(self.current_path)
        
        if letter and confidence > self.CONFIDENCE_THRESHOLD:
            self.current_word += letter
            print(f"Letter recognized: {letter} (confidence: {confidence:.3f})")
        else:
            print(f"Letter rejected: {letter} (confidence: {confidence:.3f})")
        
        # Clear current path
        self.current_path.clear()
        self.letter_pause_count = 0
    
    def process_word_completion(self):
        """Process when a word is completed"""
        if not self.current_word:
            return
        
        # Correct the word
        corrected_word = self.correct_word(self.current_word)
        
        print(f"Word completed: {self.current_word} -> {corrected_word}")
        
        # Add to recognized words
        self.recognized_words.append(corrected_word)
        
        # Speak the word
        try:
            speak_word(corrected_word)
        except Exception as e:
            print(f"Could not speak word: {e}")
        
        # Reset for next word
        self.current_word = ""
        self.word_pause_count = 0
        self.current_path.clear()
    
    def run(self):
        """Main application loop"""
        print("\n🚀 Starting Simple Word Air Writing System...")
        print("📋 Instructions:")
        print("   • Hold up your index finger (other fingers curled)")
        print("   • Write letters in the air")
        print("   • Pause briefly between letters")
        print("   • Pause longer between words")
        print("   • Try simple words like: CAT, DOG, SUN, BOX")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("❌ Failed to read from camera")
                break
            
            # Flip frame horizontally
            frame = cv2.flip(frame, 1)
            
            # Process frame with improved hand tracker
            fingertip, velocity, is_writing = self.tracker.process_frame(frame)
            
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
                
                # Check for letter completion
                if self.letter_pause_count >= self.LETTER_PAUSE_FRAMES:
                    self.process_letter_completion()
                
                # Check for word completion
                if time.time() - self.last_movement_time > 3.0:  # 3 seconds of no movement
                    self.word_pause_count += 1
                    
                    if self.word_pause_count >= self.WORD_PAUSE_FRAMES:
                        self.process_word_completion()
            
            # Draw UI
            self.draw_ui(frame)
            
            # Show frame
            cv2.imshow("Simple Word Air Writing", frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' '):  # Manual letter completion
                self.process_letter_completion()
            
            elif key == ord('s'):  # Speak current word
                if self.current_word:
                    corrected_word = self.correct_word(self.current_word)
                    self.recognized_words.append(corrected_word)
                    try:
                        speak_word(corrected_word)
                    except:
                        pass
                    self.current_word = ""
                    self.current_path.clear()
            
            elif key == ord('c'):  # Clear
                self.current_word = ""
                self.current_path.clear()
                self.tracker.clear_trail()
                print("Cleared current word")
            
            elif key == 27:  # ESC
                break
        
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()
        
        # Print session summary
        print(f"\n📊 Session Summary:")
        print(f"   Words recognized: {len(self.recognized_words)}")
        if self.recognized_words:
            print(f"   Words: {', '.join(self.recognized_words)}")
        print("👋 Simple Word Air Writing completed!")

def main():
    """Main function"""
    print("🖐️  SIMPLE WORD AIR WRITING SYSTEM")
    print("=" * 50)
    print("🎯 Optimized for simple words like CAT, DOG, SUN")
    print("=" * 50)
    
    try:
        app = SimpleWordAirWriting()
        app.run()
    except KeyboardInterrupt:
        print("\n👋 Application interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()