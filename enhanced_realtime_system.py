#!/usr/bin/env python3
"""
Enhanced Real-Time Air Writing System with Supreme Accuracy
Features: Ensemble models, test-time augmentation, advanced word prediction,
confidence-based filtering, and adaptive learning
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
import pickle

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

try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False

class EnsembleLetterRecognizer:
    """Ensemble letter recognizer with multiple models and test-time augmentation"""
    
    def __init__(self, model_paths=None):
        self.models = []
        self.model_weights = []
        self.model_available = False
        
        # Load multiple models for ensemble
        self.load_ensemble_models(model_paths)
        
        # Test-time augmentation settings
        self.tta_enabled = True
        self.tta_rotations = [-10, -5, 0, 5, 10]
        self.tta_scales = [0.9, 1.0, 1.1]
        
        # Confidence calibration
        self.confidence_history = deque(maxlen=100)
        self.calibration_factor = 1.0
        
        print(f"🧠 Ensemble recognizer initialized with {len(self.models)} models")
    
    def load_ensemble_models(self, model_paths=None):
        """Load multiple models for ensemble prediction"""
        default_paths = [
            "models/letter_recognition_supreme_optimized.h5",
            "models/supreme_optimized_letter_recognition.h5",
            "models/letter_recognition_ultra_optimized.h5",
            "models/ultra_optimized_letter_recognition.h5",
            "models/letter_recognition_advanced.h5",
            "models/letter_recognition.h5"
        ]
        
        if model_paths:
            paths_to_try = model_paths + default_paths
        else:
            paths_to_try = default_paths
        
        for path in paths_to_try:
            if os.path.exists(path):
                try:
                    model = load_model(path, compile=False)
                    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
                    
                    # Test model
                    test_input = np.zeros((1, 28, 28, 1))
                    _ = model.predict(test_input, verbose=0)
                    
                    self.models.append(model)
                    
                    # Assign weights based on model sophistication
                    if 'supreme' in path:
                        weight = 1.0
                    elif 'ultra' in path:
                        weight = 0.8
                    elif 'advanced' in path:
                        weight = 0.6
                    else:
                        weight = 0.4
                    
                    self.model_weights.append(weight)
                    print(f"✅ Loaded ensemble model: {path} (weight: {weight})")
                    
                except Exception as e:
                    print(f"⚠️  Failed to load {path}: {e}")
        
        if self.models:
            self.model_available = True
            # Normalize weights
            total_weight = sum(self.model_weights)
            self.model_weights = [w / total_weight for w in self.model_weights]
        else:
            print("⚠️  No models loaded - using demo mode")
    
    def apply_test_time_augmentation(self, image):
        """Apply test-time augmentation to improve accuracy"""
        augmented_images = []
        
        # Original image
        augmented_images.append(image)
        
        if self.tta_enabled:
            h, w = image.shape[:2]
            center = (w // 2, h // 2)
            
            # Rotations
            for angle in self.tta_rotations[1:]:  # Skip 0 as it's already added
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                rotated = cv2.warpAffine(image, M, (w, h), borderValue=255)
                augmented_images.append(rotated)
            
            # Scales
            for scale in self.tta_scales:
                if scale != 1.0:
                    new_size = int(min(w, h) * scale)
                    scaled = cv2.resize(image, (new_size, new_size))
                    
                    # Pad or crop to original size
                    if new_size < min(w, h):
                        # Pad
                        pad_size = (min(w, h) - new_size) // 2
                        scaled = cv2.copyMakeBorder(scaled, pad_size, pad_size, pad_size, pad_size, 
                                                  cv2.BORDER_CONSTANT, value=255)
                    else:
                        # Crop
                        start = (new_size - min(w, h)) // 2
                        scaled = scaled[start:start+min(w, h), start:start+min(w, h)]
                    
                    scaled = cv2.resize(scaled, (w, h))
                    augmented_images.append(scaled)
        
        return augmented_images
    
    def ensemble_predict(self, image):
        """Predict using ensemble of models with TTA"""
        if not self.model_available:
            return self.demo_recognition()
        
        # Apply test-time augmentation
        augmented_images = self.apply_test_time_augmentation(image)
        
        # Collect predictions from all models and augmentations
        all_predictions = []
        
        for model, weight in zip(self.models, self.model_weights):
            model_predictions = []
            
            for aug_image in augmented_images:
                # Preprocess for model
                processed = cv2.resize(aug_image, (28, 28))
                processed = processed.astype(np.float32) / 255.0
                processed = processed.reshape(1, 28, 28, 1)
                
                # Predict
                pred = model.predict(processed, verbose=0)[0]
                model_predictions.append(pred)
            
            # Average predictions across augmentations
            avg_pred = np.mean(model_predictions, axis=0)
            
            # Weight by model importance
            weighted_pred = avg_pred * weight
            all_predictions.append(weighted_pred)
        
        # Ensemble prediction
        final_prediction = np.sum(all_predictions, axis=0)
        
        # Get letter and confidence
        letter_idx = np.argmax(final_prediction)
        confidence = final_prediction[letter_idx]
        
        # Apply confidence calibration
        calibrated_confidence = self.calibrate_confidence(confidence)
        
        letter = chr(letter_idx + ord('A'))
        
        # Update confidence history
        self.confidence_history.append(calibrated_confidence)
        
        return letter, calibrated_confidence
    
    def calibrate_confidence(self, raw_confidence):
        """Calibrate confidence based on historical performance"""
        if len(self.confidence_history) < 10:
            return raw_confidence
        
        # Simple calibration based on historical average
        historical_avg = np.mean(self.confidence_history)
        
        # Adjust calibration factor
        if historical_avg > 0.8:
            self.calibration_factor = min(1.2, self.calibration_factor + 0.01)
        elif historical_avg < 0.5:
            self.calibration_factor = max(0.8, self.calibration_factor - 0.01)
        
        return min(1.0, raw_confidence * self.calibration_factor)
    
    def demo_recognition(self):
        """Demo recognition for when no models are available"""
        letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        letter = np.random.choice(list(letters))
        confidence = np.random.uniform(0.4, 0.8)
        return letter, confidence

class AdvancedWordPredictor:
    """Advanced word prediction with context and pattern matching"""
    
    def __init__(self):
        self.dictionary_data = self.load_advanced_dictionary()
        self.context_history = deque(maxlen=5)
        self.letter_sequence_patterns = {}
        self.word_transition_matrix = {}
        
        if self.dictionary_data:
            self.target_words = self.dictionary_data['target_words']
            self.letter_frequencies = self.dictionary_data.get('letter_frequencies', {})
            self.bigram_frequencies = self.dictionary_data.get('bigram_frequencies', {})
            self.position_patterns = self.dictionary_data.get('position_patterns', {})
            self.similarity_matrix = self.dictionary_data.get('similarity_matrix', {})
            
            # Build transition matrix
            self.build_word_transition_matrix()
            
            print(f"📚 Advanced predictor loaded with {len(self.target_words)} words")
        else:
            self.target_words = ['CAT', 'DOG', 'SUN', 'BOX', 'RED', 'BIG', 'TOP', 'CUP']
            self.letter_frequencies = {}
            self.bigram_frequencies = {}
            self.position_patterns = {}
            self.similarity_matrix = {}
            print("📚 Using basic word predictor")
        
        self.word_set = set(self.target_words)
    
    def load_advanced_dictionary(self):
        """Load advanced dictionary with all features"""
        dict_paths = [
            "models/supreme_optimized_word_dictionary.json",
            "models/ultra_optimized_word_dictionary.json"
        ]
        
        for path in dict_paths:
            if os.path.exists(path):
                try:
                    with open(path, 'r') as f:
                        return json.load(f)
                except Exception as e:
                    print(f"⚠️  Failed to load {path}: {e}")
        
        return None
    
    def build_word_transition_matrix(self):
        """Build word transition probabilities for context prediction"""
        # Simple bigram model for word sequences
        for i in range(len(self.target_words) - 1):
            word1 = self.target_words[i]
            word2 = self.target_words[i + 1]
            
            if word1 not in self.word_transition_matrix:
                self.word_transition_matrix[word1] = {}
            
            if word2 not in self.word_transition_matrix[word1]:
                self.word_transition_matrix[word1][word2] = 0
            
            self.word_transition_matrix[word1][word2] += 1
        
        # Normalize probabilities
        for word1 in self.word_transition_matrix:
            total = sum(self.word_transition_matrix[word1].values())
            for word2 in self.word_transition_matrix[word1]:
                self.word_transition_matrix[word1][word2] /= total
    
    def predict_next_words(self, current_context):
        """Predict likely next words based on context"""
        if not current_context or not self.word_transition_matrix:
            return []
        
        last_word = current_context[-1]
        
        if last_word in self.word_transition_matrix:
            predictions = sorted(
                self.word_transition_matrix[last_word].items(),
                key=lambda x: x[1],
                reverse=True
            )
            return [word for word, prob in predictions[:5]]
        
        return []
    
    def advanced_word_correction(self, partial_word, letter_confidences=None):
        """Advanced word correction using multiple signals"""
        if not partial_word:
            return ""
        
        partial_word = partial_word.upper().strip()
        
        # Direct match
        if partial_word in self.word_set:
            return partial_word
        
        # Calculate scores for all candidate words
        candidates = []
        
        for target_word in self.target_words:
            score = self.calculate_word_score(partial_word, target_word, letter_confidences)
            candidates.append((target_word, score))
        
        # Sort by score
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Context-based adjustment
        if self.context_history:
            context_predictions = self.predict_next_words(list(self.context_history))
            
            # Boost scores for contextually likely words
            for i, (word, score) in enumerate(candidates):
                if word in context_predictions:
                    context_boost = 1.0 + (0.2 * (5 - context_predictions.index(word)))
                    candidates[i] = (word, score * context_boost)
            
            # Re-sort after context adjustment
            candidates.sort(key=lambda x: x[1], reverse=True)
        
        return candidates[0][0] if candidates else partial_word
    
    def calculate_word_score(self, partial_word, target_word, letter_confidences=None):
        """Calculate comprehensive word matching score"""
        if not partial_word or not target_word:
            return 0.0
        
        # Length penalty
        length_diff = abs(len(partial_word) - len(target_word))
        length_penalty = max(0, 1.0 - length_diff * 0.2)
        
        # Character similarity
        char_score = self.character_similarity(partial_word, target_word)
        
        # Position-based similarity
        position_score = self.positional_similarity(partial_word, target_word)
        
        # Pattern-based score using position patterns
        pattern_score = self.pattern_similarity(partial_word, target_word)
        
        # Confidence-weighted score
        confidence_score = 1.0
        if letter_confidences and len(letter_confidences) == len(partial_word):
            confidence_score = np.mean(letter_confidences)
        
        # Frequency-based score
        frequency_score = self.frequency_similarity(partial_word, target_word)
        
        # Combined score
        total_score = (
            char_score * 0.3 +
            position_score * 0.25 +
            pattern_score * 0.2 +
            frequency_score * 0.15 +
            length_penalty * 0.1
        ) * confidence_score
        
        return total_score
    
    def character_similarity(self, word1, word2):
        """Character-based similarity"""
        if not word1 or not word2:
            return 0.0
        
        matches = sum(1 for a, b in zip(word1, word2) if a == b)
        max_length = max(len(word1), len(word2))
        return matches / max_length if max_length > 0 else 0.0
    
    def positional_similarity(self, word1, word2):
        """Position-weighted similarity"""
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
    
    def pattern_similarity(self, word1, target_word):
        """Pattern-based similarity using position patterns"""
        if not self.position_patterns or not word1:
            return 0.5
        
        score = 0.0
        total_positions = 0
        
        for i, letter in enumerate(word1):
            if str(i) in self.position_patterns:
                pattern_freq = self.position_patterns[str(i)]
                if letter in pattern_freq and i < len(target_word):
                    if target_word[i] == letter:
                        score += pattern_freq[letter] / sum(pattern_freq.values())
                    total_positions += 1
        
        return score / max(total_positions, 1)
    
    def frequency_similarity(self, word1, target_word):
        """Frequency-based similarity"""
        if not self.letter_frequencies:
            return 0.5
        
        score = 0.0
        for letter in word1:
            if letter in self.letter_frequencies:
                score += self.letter_frequencies[letter]
        
        target_score = 0.0
        for letter in target_word:
            if letter in self.letter_frequencies:
                target_score += self.letter_frequencies[letter]
        
        if target_score > 0:
            return min(1.0, score / target_score)
        
        return 0.5
    
    def update_context(self, word):
        """Update context history"""
        if word and word in self.word_set:
            self.context_history.append(word)
    
    def get_smart_suggestions(self, partial_word):
        """Get smart word suggestions"""
        if not partial_word:
            # Return contextually likely words
            if self.context_history:
                return self.predict_next_words(list(self.context_history))
            return self.target_words[:5]
        
        # Get words that start with partial input
        prefix_matches = [word for word in self.target_words if word.startswith(partial_word.upper())]
        
        if prefix_matches:
            return prefix_matches[:5]
        
        # Get similar words
        candidates = []
        for word in self.target_words:
            score = self.calculate_word_score(partial_word, word)
            candidates.append((word, score))
        
        candidates.sort(key=lambda x: x[1], reverse=True)
        return [word for word, score in candidates[:5]]

class EnhancedAirWritingSystem:
    """Enhanced air writing system with supreme accuracy"""
    
    def __init__(self, model_paths=None):
        print("🚀 Initializing Enhanced Air Writing System...")
        
        # Initialize enhanced components
        from complete_airwriting_system import UniversalHandTracker, TextToSpeech
        
        self.hand_tracker = UniversalHandTracker()
        self.letter_recognizer = EnsembleLetterRecognizer(model_paths)
        self.word_predictor = AdvancedWordPredictor()
        self.tts = TextToSpeech()
        
        # Initialize camera with optimal settings
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise Exception("Could not open camera")
        
        # Optimized camera settings
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Enhanced state management
        self.current_word = ""
        self.recognized_words = []
        self.current_path = []
        self.letter_candidates = []
        self.letter_confidences = []
        
        # Adaptive parameters
        self.LETTER_PAUSE_FRAMES = 15
        self.WORD_PAUSE_FRAMES = 45
        self.MIN_PATH_LENGTH = 6
        self.CONFIDENCE_THRESHOLD = 0.2  # Lower threshold with ensemble
        self.HIGH_CONFIDENCE_THRESHOLD = 0.8
        
        # Performance tracking
        self.fps = 0
        self.frame_count = 0
        self.fps_start_time = time.time()
        self.processing_times = deque(maxlen=30)
        
        # Enhanced features
        self.adaptive_thresholds = True
        self.smart_word_completion = True
        self.context_aware_prediction = True
        self.confidence_based_filtering = True
        
        # State counters
        self.letter_pause_count = 0
        self.word_pause_count = 0
        self.last_movement_time = time.time()
        self.last_letter_time = time.time()
        
        print("✅ Enhanced Air Writing System initialized")
        self.print_enhanced_instructions()
    
    def print_enhanced_instructions(self):
        """Print enhanced usage instructions"""
        print("\n" + "="*70)
        print("🖐️  ENHANCED AIR WRITING SYSTEM - SUPREME ACCURACY")
        print("="*70)
        print("📋 Features:")
        print("   • Ensemble model prediction with test-time augmentation")
        print("   • Context-aware word prediction")
        print("   • Adaptive confidence thresholds")
        print("   • Smart auto-completion")
        print("\n🎯 Try these words:")
        sample_words = self.word_predictor.target_words[:12]
        for i in range(0, len(sample_words), 4):
            print("   " + " | ".join(sample_words[i:i+4]))
        print("\n⌨️  Controls:")
        print("   SPACE - End current letter    S - Speak word")
        print("   C     - Clear current word    A - Toggle auto-complete")
        print("   T     - Toggle TTA           D - Toggle debug")
        print("   ESC   - Exit system")
        print("="*70 + "\n")
    
    def process_enhanced_letter_completion(self):
        """Enhanced letter processing with ensemble prediction"""
        if len(self.current_path) < self.MIN_PATH_LENGTH:
            return
        
        processing_start = time.time()
        
        # Create image from path
        img = self.create_letter_image(self.current_path)
        
        # Ensemble prediction
        letter, confidence = self.letter_recognizer.ensemble_predict(img)
        
        processing_time = (time.time() - processing_start) * 1000
        self.processing_times.append(processing_time)
        
        # Adaptive confidence threshold
        if self.adaptive_thresholds:
            threshold = self.calculate_adaptive_threshold()
        else:
            threshold = self.CONFIDENCE_THRESHOLD
        
        if letter and confidence > threshold:
            # Store letter with confidence
            self.letter_candidates.append((letter, confidence, time.time()))
            self.letter_confidences.append(confidence)
            
            # Apply letter smoothing with confidence weighting
            final_letter = self.apply_confidence_smoothing(letter, confidence)
            
            # Add to current word
            self.current_word += final_letter
            self.last_letter_time = time.time()
            
            print(f"Letter: {final_letter} (confidence: {confidence:.3f}, threshold: {threshold:.3f})")
            
            # Smart auto-completion
            if self.smart_word_completion:
                self.check_smart_completion()
        else:
            print(f"Letter rejected: {letter} (confidence: {confidence:.3f}, threshold: {threshold:.3f})")
        
        # Clear path
        self.current_path.clear()
        self.letter_pause_count = 0
    
    def calculate_adaptive_threshold(self):
        """Calculate adaptive confidence threshold"""
        if len(self.letter_confidences) < 5:
            return self.CONFIDENCE_THRESHOLD
        
        recent_confidences = self.letter_confidences[-10:]
        avg_confidence = np.mean(recent_confidences)
        std_confidence = np.std(recent_confidences)
        
        # Adaptive threshold based on recent performance
        adaptive_threshold = max(
            self.CONFIDENCE_THRESHOLD,
            avg_confidence - 2 * std_confidence
        )
        
        return min(adaptive_threshold, 0.6)  # Cap at 0.6
    
    def apply_confidence_smoothing(self, letter, confidence):
        """Apply confidence-weighted letter smoothing"""
        if len(self.letter_candidates) < 2:
            return letter
        
        # Get recent candidates
        recent_candidates = self.letter_candidates[-3:]
        
        # Weighted voting based on confidence
        letter_votes = {}
        total_weight = 0
        
        for candidate_letter, candidate_confidence, _ in recent_candidates:
            if candidate_letter not in letter_votes:
                letter_votes[candidate_letter] = 0
            letter_votes[candidate_letter] += candidate_confidence
            total_weight += candidate_confidence
        
        # Normalize votes
        if total_weight > 0:
            for l in letter_votes:
                letter_votes[l] /= total_weight
        
        # Return letter with highest weighted vote
        if letter_votes:
            best_letter = max(letter_votes.items(), key=lambda x: x[1])
            if best_letter[1] > 0.4:  # Minimum vote threshold
                return best_letter[0]
        
        return letter
    
    def check_smart_completion(self):
        """Smart word completion with context awareness"""
        if len(self.current_word) < 2:
            return
        
        # Get suggestions from advanced predictor
        suggestions = self.word_predictor.get_smart_suggestions(self.current_word)
        
        # Check for high-confidence exact matches
        exact_matches = [word for word in suggestions if word.startswith(self.current_word)]
        
        if len(exact_matches) == 1 and len(self.current_word) >= 3:
            # High confidence single match
            if len(self.letter_confidences) >= 2:
                avg_confidence = np.mean(self.letter_confidences[-len(self.current_word):])
                if avg_confidence > self.HIGH_CONFIDENCE_THRESHOLD:
                    completed_word = exact_matches[0]
                    if len(completed_word) > len(self.current_word):
                        print(f"🤖 Auto-completing: {self.current_word} → {completed_word}")
                        self.current_word = completed_word
                        self.process_enhanced_word_completion()
    
    def process_enhanced_word_completion(self):
        """Enhanced word completion with context update"""
        if not self.current_word:
            return
        
        # Advanced word correction
        corrected_word = self.word_predictor.advanced_word_correction(
            self.current_word, 
            self.letter_confidences[-len(self.current_word):] if self.letter_confidences else None
        )
        
        # Calculate word confidence
        word_confidence = np.mean(self.letter_confidences[-len(self.current_word):]) if self.letter_confidences else 0.5
        
        print(f"Word: {self.current_word} → {corrected_word} (confidence: {word_confidence:.3f})")
        
        # Add to recognized words
        self.recognized_words.append(corrected_word)
        
        # Update context
        self.word_predictor.update_context(corrected_word)
        
        # Speak word
        threading.Thread(target=self.tts.speak, args=(corrected_word,), daemon=True).start()
        
        # Reset state
        self.reset_enhanced_state()
    
    def create_letter_image(self, path):
        """Create optimized letter image from path"""
        if len(path) < 2:
            return np.ones((32, 32), dtype=np.uint8) * 255
        
        # Use enhanced preprocessing
        from utils.preprocessing import draw_path_on_blank, enhance_letter_image
        
        img = draw_path_on_blank(path, img_size=32)
        enhanced_img = enhance_letter_image(img)
        
        return enhanced_img
    
    def reset_enhanced_state(self):
        """Reset enhanced state"""
        self.current_word = ""
        self.word_pause_count = 0
        self.current_path.clear()
        self.letter_candidates.clear()
        self.letter_confidences.clear()
    
    def draw_enhanced_ui(self, frame):
        """Draw enhanced UI with more information"""
        h, w = frame.shape[:2]
        
        # Main panel
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (w-10, 200), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        
        # Current word with confidence indicator
        word_text = f"Word: {self.current_word}"
        word_color = (0, 255, 255) if self.current_word else (128, 128, 128)
        cv2.putText(frame, word_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, word_color, 3)
        
        # Word confidence bar
        if self.letter_confidences and self.current_word:
            avg_confidence = np.mean(self.letter_confidences[-len(self.current_word):])
            bar_width = int(300 * avg_confidence)
            cv2.rectangle(frame, (20, 60), (20 + bar_width, 70), (0, 255, 0), -1)
            cv2.rectangle(frame, (20, 60), (320, 70), (255, 255, 255), 2)
            cv2.putText(frame, f"Confidence: {avg_confidence:.2f}", (330, 68), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Smart suggestions
        suggestions = self.word_predictor.get_smart_suggestions(self.current_word)
        if suggestions:
            suggestion_text = f"Suggestions: {' | '.join(suggestions[:4])}"
            cv2.putText(frame, suggestion_text, (20, 95), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Context history
        if self.word_predictor.context_history:
            context_text = f"Context: {' → '.join(list(self.word_predictor.context_history)[-3:])}"
            cv2.putText(frame, context_text, (20, 125), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        
        # Recent words
        if self.recognized_words:
            recent_text = f"Recent: {' → '.join(self.recognized_words[-4:])}"
            cv2.putText(frame, recent_text, (20, 155), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # System status
        status_y = 185
        status_items = [
            f"Models: {len(self.letter_recognizer.models)}",
            f"TTA: {'ON' if self.letter_recognizer.tta_enabled else 'OFF'}",
            f"Adaptive: {'ON' if self.adaptive_thresholds else 'OFF'}",
            f"Auto: {'ON' if self.smart_word_completion else 'OFF'}"
        ]
        
        for i, status in enumerate(status_items):
            cv2.putText(frame, status, (20 + i * 150, status_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Performance metrics
        avg_processing = np.mean(self.processing_times) if self.processing_times else 0
        cv2.putText(frame, f"FPS: {self.fps} | Processing: {avg_processing:.1f}ms", 
                   (w - 300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Path info
        if self.current_path:
            path_info = f"Path: {len(self.current_path)} points"
            cv2.putText(frame, path_info, (w - 300, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def update_fps(self):
        """Update FPS counter"""
        self.frame_count += 1
        if time.time() - self.fps_start_time >= 1.0:
            self.fps = self.frame_count
            self.frame_count = 0
            self.fps_start_time = time.time()
    
    def run_enhanced_system(self):
        """Run the enhanced air writing system"""
        print("🚀 Starting Enhanced Air Writing System...")
        
        while True:
            frame_start = time.time()
            
            ret, frame = self.cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            
            # Process hand tracking
            fingertip, velocity, is_writing = self.hand_tracker.process_frame(frame)
            
            # Update FPS
            self.update_fps()
            
            # Enhanced writing logic
            current_time = time.time()
            
            if fingertip and is_writing:
                self.current_path.append(fingertip)
                self.last_movement_time = current_time
                self.letter_pause_count = 0
                self.word_pause_count = 0
            else:
                if len(self.current_path) > 0:
                    self.letter_pause_count += 1
                
                # Letter completion
                if (self.letter_pause_count >= self.LETTER_PAUSE_FRAMES or
                    (len(self.current_path) >= self.MIN_PATH_LENGTH and 
                     current_time - self.last_movement_time > 0.8)):
                    self.process_enhanced_letter_completion()
                
                # Word completion
                time_since_movement = current_time - self.last_movement_time
                time_since_letter = current_time - self.last_letter_time
                
                if (time_since_movement > 2.0 or 
                    (self.current_word and time_since_letter > 3.5)):
                    self.word_pause_count += 1
                    
                    if (self.word_pause_count >= self.WORD_PAUSE_FRAMES or
                        time_since_movement > 4.0):
                        self.process_enhanced_word_completion()
            
            # Draw enhanced UI
            self.draw_enhanced_ui(frame)
            
            # Show frame
            cv2.imshow("Enhanced Air Writing System", frame)
            
            # Handle input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' '):
                self.process_enhanced_letter_completion()
            elif key == ord('s'):
                if self.current_word:
                    self.process_enhanced_word_completion()
            elif key == ord('c'):
                self.reset_enhanced_state()
                self.hand_tracker.clear_trail()
            elif key == ord('a'):
                self.smart_word_completion = not self.smart_word_completion
                print(f"Auto-completion: {'ON' if self.smart_word_completion else 'OFF'}")
            elif key == ord('t'):
                self.letter_recognizer.tta_enabled = not self.letter_recognizer.tta_enabled
                print(f"Test-time augmentation: {'ON' if self.letter_recognizer.tta_enabled else 'OFF'}")
            elif key == ord('d'):
                self.adaptive_thresholds = not self.adaptive_thresholds
                print(f"Adaptive thresholds: {'ON' if self.adaptive_thresholds else 'OFF'}")
            elif key == 27:
                break
        
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()
        
        print(f"\n📊 Enhanced Session Summary:")
        print(f"   Words: {' → '.join(self.recognized_words)}")
        print(f"   Average processing: {np.mean(self.processing_times):.1f}ms")
        print("👋 Enhanced system finished!")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Enhanced Air Writing System")
    parser.add_argument('--models', nargs='+', help='Paths to model files')
    args = parser.parse_args()
    
    try:
        system = EnhancedAirWritingSystem(model_paths=args.models)
        system.run_enhanced_system()
    except KeyboardInterrupt:
        print("\n👋 System interrupted")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()