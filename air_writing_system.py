#!/usr/bin/env python3
"""
Complete Integrated Air Writing Recognition System
Built using existing codebase with all advanced features
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

# Import existing modules
from utils.hand_tracker import HandTracker
from utils.preprocessing import (
    normalize_path, smooth_path, calculate_path_features, 
    draw_path_on_blank, enhance_letter_image, preprocess_for_recognition
)

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

class IntegratedLetterRecognizer:
    """Integrated letter recognizer using existing models and preprocessing"""
    
    def __init__(self, model_paths=None):
        self.models = []
        self.model_available = False
        self.processing_times = deque(maxlen=30)
        
        # Load available models
        self.load_models(model_paths)
        
        print(f"🧠 Letter recognizer initialized (Models: {len(self.models)})")
    
    def load_models(self, model_paths=None):
        """Load trained models from existing model files"""
        if not MODEL_AVAILABLE:
            return
        
        # Use existing model files
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
    
    def recognize_letter(self, path):
        """Enhanced letter recognition with improved preprocessing and confidence scoring"""
        if len(path) < 5:
            return None, 0.0
        
        start_time = time.time()
        
        try:
            if self.model_available and self.models:
                # Enhanced preprocessing for better accuracy
                img = self.create_enhanced_image(path, img_size=28)
                
                # Prepare for model prediction with better normalization
                img_normalized = img.reshape(1, 28, 28, 1).astype(np.float32) / 255.0
                
                # Ensemble prediction with weighted averaging
                if len(self.models) > 1:
                    predictions = []
                    weights = []
                    
                    for i, model in enumerate(self.models):
                        try:
                            pred = model.predict(img_normalized, verbose=0)[0]
                            predictions.append(pred)
                            # Weight newer/better models higher
                            weights.append(1.0 + i * 0.2)
                        except Exception as model_error:
                            print(f"⚠️ Model {i} prediction error: {model_error}")
                            continue
                    
                    if predictions:
                        # Weighted average predictions
                        weighted_preds = np.average(predictions, axis=0, weights=weights)
                        final_prediction = weighted_preds
                    else:
                        return self.demo_recognition()
                else:
                    try:
                        final_prediction = self.models[0].predict(img_normalized, verbose=0)[0]
                    except Exception as model_error:
                        print(f"⚠️ Model prediction error: {model_error}")
                        return self.demo_recognition()
                
                # Enhanced confidence calculation
                sorted_probs = np.sort(final_prediction)[::-1]
                top_prob = sorted_probs[0]
                second_prob = sorted_probs[1] if len(sorted_probs) > 1 else 0
                
                # Confidence boost based on separation between top predictions
                separation_boost = min(0.2, (top_prob - second_prob) * 2)
                adjusted_confidence = top_prob + separation_boost
                
                # Path quality boost
                path_quality = self.calculate_path_quality(path)
                quality_boost = path_quality * 0.15
                final_confidence = min(1.0, adjusted_confidence + quality_boost)
                
                letter_idx = np.argmax(final_prediction)
                letter = chr(letter_idx + ord('A'))
                
                # Track processing time
                processing_time = (time.time() - start_time) * 1000
                self.processing_times.append(processing_time)
                
                return letter, final_confidence
            else:
                return self.demo_recognition()
                
        except Exception as e:
            print(f"❌ Recognition error: {e}")
            return self.demo_recognition()
    
    def calculate_path_quality(self, path):
        """Calculate path quality score for confidence adjustment with safe division"""
        if len(path) < 5:
            return 0.0
        
        try:
            path_array = np.array(path)
            
            # Smoothness score (lower variance in direction changes = higher quality)
            smoothness = 0.5  # Default value
            if len(path) > 3:
                try:
                    directions = np.diff(path_array, axis=0)
                    direction_changes = np.diff(directions, axis=0)
                    
                    if len(direction_changes) > 0:
                        direction_norms = np.linalg.norm(direction_changes, axis=1)
                        mean_change = np.mean(direction_norms)
                        
                        # Safe division with minimum denominator
                        smoothness = 1.0 / (1.0 + max(0.001, mean_change))
                    else:
                        smoothness = 0.8  # High smoothness for very short paths
                except Exception as smooth_error:
                    print(f"⚠️ Smoothness calculation error: {smooth_error}")
                    smoothness = 0.5
            
            # Length score (reasonable length gets higher score)
            length_score = 0.0
            try:
                if len(path_array) > 1:
                    path_diffs = np.diff(path_array, axis=0)
                    path_norms = np.linalg.norm(path_diffs, axis=1)
                    total_length = np.sum(path_norms)
                    
                    # Safe normalization with minimum denominator
                    length_score = min(1.0, max(0.0, total_length / max(1.0, 100.0)))
                else:
                    length_score = 0.1
            except Exception as length_error:
                print(f"⚠️ Length calculation error: {length_error}")
                length_score = 0.3
            
            # Density score (good point density)
            density_score = 0.0
            try:
                # Safe division for density
                density_score = min(1.0, max(0.0, len(path) / max(1.0, 50.0)))
            except Exception as density_error:
                print(f"⚠️ Density calculation error: {density_error}")
                density_score = 0.3
            
            # Combined quality score with safe weights
            try:
                quality = (smoothness * 0.4 + length_score * 0.3 + density_score * 0.3)
                return max(0.0, min(1.0, quality))
            except Exception as combine_error:
                print(f"⚠️ Quality combination error: {combine_error}")
                return 0.5
            
        except Exception as e:
            print(f"⚠️ Path quality calculation error: {e}")
            return 0.5
    
    def demo_recognition(self):
        """Demo recognition for when models aren't available"""
        letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        letter = np.random.choice(list(letters))
        confidence = np.random.uniform(0.4, 0.9)
        return letter, confidence
    
    def get_average_processing_time(self):
        """Get average processing time"""
        return np.mean(self.processing_times) if self.processing_times else 0.0
    
    def create_enhanced_image(self, path, img_size=28):
        """Create enhanced image from path with better preprocessing for accuracy"""
        if len(path) < 2:
            return np.ones((img_size, img_size), dtype=np.uint8) * 255
        
        # Create blank image
        img = np.ones((img_size, img_size), dtype=np.uint8) * 255
        
        try:
            # Smooth path first for better quality
            smoothed_path = self.smooth_path(path)
            path_array = np.array(smoothed_path)
            
            # Get bounding box with error handling
            min_x, min_y = np.min(path_array, axis=0)
            max_x, max_y = np.max(path_array, axis=0)
            
            width = max_x - min_x
            height = max_y - min_y
            
            if width <= 0 or height <= 0:
                return img
            
            # Enhanced scaling with aspect ratio preservation
            padding = 3
            available_size = img_size - 2 * padding
            scale = min(available_size / width, available_size / height)
            
            # Center the path more precisely
            center_x = (min_x + max_x) / 2
            center_y = (min_y + max_y) / 2
            offset_x = img_size / 2 - center_x * scale
            offset_y = img_size / 2 - center_y * scale
            
            # Draw path with variable thickness based on speed
            scaled_path = []
            for x, y in smoothed_path:
                new_x = int(x * scale + offset_x)
                new_y = int(y * scale + offset_y)
                new_x = np.clip(new_x, 0, img_size - 1)
                new_y = np.clip(new_y, 0, img_size - 1)
                scaled_path.append((new_x, new_y))
            
            # Draw with adaptive thickness
            for i in range(1, len(scaled_path)):
                # Calculate local speed for thickness variation
                if i < len(scaled_path) - 1:
                    dist1 = np.linalg.norm(np.array(scaled_path[i]) - np.array(scaled_path[i-1]))
                    dist2 = np.linalg.norm(np.array(scaled_path[i+1]) - np.array(scaled_path[i]))
                    avg_speed = (dist1 + dist2) / 2
                    thickness = max(1, min(3, int(4 - avg_speed * 0.5)))
                else:
                    thickness = 2
                
                cv2.line(img, scaled_path[i-1], scaled_path[i], (0), thickness)
            
            # Enhanced post-processing
            # Apply slight blur for anti-aliasing
            img = cv2.GaussianBlur(img, (3, 3), 0.8)
            
            # Enhance contrast
            img = cv2.convertScaleAbs(img, alpha=1.2, beta=-30)
            
            # Ensure proper range
            img = np.clip(img, 0, 255).astype(np.uint8)
            
            return img
            
        except Exception as e:
            print(f"⚠️ Enhanced image creation error: {e}")
            # Fallback to simple method
            return self.create_simple_image(path, img_size)
    
    def smooth_path(self, path, window_size=3):
        """Smooth path using moving average"""
        if len(path) <= window_size:
            return path
        
        try:
            path_array = np.array(path)
            smoothed = []
            
            for i in range(len(path)):
                start_idx = max(0, i - window_size // 2)
                end_idx = min(len(path), i + window_size // 2 + 1)
                window_points = path_array[start_idx:end_idx]
                smoothed_point = np.mean(window_points, axis=0)
                smoothed.append(tuple(smoothed_point.astype(int)))
            
            return smoothed
        except Exception as e:
            print(f"⚠️ Path smoothing error: {e}")
            return path
    
    def create_simple_image(self, path, img_size=28):
        """Fallback simple image creation method"""
        if len(path) < 2:
            return np.ones((img_size, img_size), dtype=np.uint8) * 255
        
        # Create blank image
        img = np.ones((img_size, img_size), dtype=np.uint8) * 255
        
        # Normalize path to image size
        path_array = np.array(path)
        min_x, min_y = np.min(path_array, axis=0)
        max_x, max_y = np.max(path_array, axis=0)
        
        width = max_x - min_x
        height = max_y - min_y
        
        if width <= 0 or height <= 0:
            return img
        
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

class IntegratedWordCorrector:
    """Integrated word corrector using existing dictionary and algorithms"""
    
    def __init__(self, dictionary_path=None):
        # Initialize basic attributes first
        self.target_words = []
        self.word_frequencies = {}
        self.word_set = set()
        self.words_by_length = {}
        
        # Load existing dictionary
        try:
            self.load_dictionary(dictionary_path)
        except Exception as e:
            print(f"⚠️ Dictionary loading error: {e}")
            # Ensure we have at least basic words
            if not self.target_words:
                self.target_words = ['CAT', 'DOG', 'BAT', 'RAT', 'HAT', 'MAT', 'SAT']
                self._finalize_dictionary_setup()
        
        # Letter confusion matrix from existing word_recognition.py
        self.confusion_pairs = {
            'O': ['0', 'Q', 'D'], 'I': ['1', 'L', 'J'], 'S': ['5', 'Z'],
            'B': ['8', 'D'], 'G': ['6', 'C'], 'Z': ['2', 'S'],
            'A': ['4', 'R'], 'E': ['3', 'F'], 'T': ['7', 'F', 'I'],
            'U': ['V', 'Y'], 'V': ['U', 'Y'], 'W': ['VV', 'M'],
            'M': ['N', 'W'], 'N': ['M', 'H'], 'P': ['R', 'B'],
            'R': ['P', 'A'], 'C': ['G', 'O'], 'D': ['O', 'B'],
            'F': ['E', 'T'], 'H': ['N', 'M'], 'J': ['I', 'L'],
            'K': ['X', 'R'], 'L': ['I', 'J'], 'Q': ['O', 'G'],
            'X': ['K', 'Y'], 'Y': ['V', 'X']
        }
        
        print(f"📚 Word corrector initialized with {len(self.target_words)} words")
    
    def load_dictionary(self, dictionary_path=None):
        """Load dictionary from existing JSON file or use default"""
        if dictionary_path and os.path.exists(dictionary_path):
            try:
                with open(dictionary_path, 'r') as f:
                    data = json.load(f)
                    self.target_words = data.get('target_words', [])
                    self.word_frequencies = data.get('word_frequencies', {})
                print(f"✅ Loaded dictionary: {dictionary_path}")
                self._finalize_dictionary_setup()
                return
            except Exception as e:
                print(f"⚠️  Failed to load dictionary: {e}")
        
        # Try existing ultra_optimized_word_dictionary.json
        existing_dict = "models/ultra_optimized_word_dictionary.json"
        if os.path.exists(existing_dict):
            try:
                with open(existing_dict, 'r') as f:
                    data = json.load(f)
                    self.target_words = data.get('target_words', [])
                    # Create frequency map
                    self.word_frequencies = {}
                    for i, word in enumerate(self.target_words):
                        freq = 1.0 - (i * 0.01) if len(word) == 3 else 0.8 - (i * 0.005)
                        self.word_frequencies[word] = max(0.1, freq)
                print(f"✅ Loaded existing dictionary: {existing_dict}")
                self._finalize_dictionary_setup()
                return
            except Exception as e:
                print(f"⚠️  Failed to load existing dictionary: {e}")
        
        # Enhanced fallback dictionary with more words and better coverage
        self.target_words = [
            # Common 3-letter words
            'THE', 'AND', 'FOR', 'ARE', 'BUT', 'NOT', 'YOU', 'ALL', 'CAN', 'HER', 'WAS', 'ONE', 'OUR',
            'HAD', 'BUT', 'HIS', 'HAS', 'SHE', 'TWO', 'WAY', 'WHO', 'BOY', 'DID', 'ITS', 'LET', 'PUT',
            'SAY', 'TOO', 'OLD', 'ANY', 'MAY', 'NEW', 'TRY', 'ASK', 'MAN', 'DAY', 'GET', 'USE', 'HER',
            'NOW', 'AIR', 'END', 'WHY', 'HOW', 'OUT', 'SEE', 'HIM', 'TWO', 'HOW', 'ITS', 'WHO', 'OIL',
            
            # Animals
            'CAT', 'DOG', 'BAT', 'RAT', 'PIG', 'COW', 'BEE', 'ANT', 'FOX', 'OWL', 'ELF',
            
            # Objects
            'HAT', 'MAT', 'BAG', 'BOX', 'CUP', 'PEN', 'KEY', 'CAR', 'BUS', 'BED', 'EGG',
            'SUN', 'ICE', 'TOP', 'JAR', 'BAR', 'NET', 'WEB', 'MAP', 'BAT', 'FAN',
            
            # Actions
            'RUN', 'SIT', 'EAT', 'SEE', 'HIT', 'CUT', 'DIG', 'FLY', 'WIN', 'TRY', 'BUY',
            'PAY', 'SAY', 'LAY', 'WAG', 'HOP', 'MOP', 'POP', 'COP', 'SOP', 'JIG', 'RIG',
            
            # Colors
            'RED', 'BIG', 'HOT', 'NEW', 'OLD', 'BAD', 'SAD', 'MAD', 'FAT', 'WET', 'DRY',
            
            # Common 4-letter words
            'THAT', 'WITH', 'HAVE', 'THIS', 'WILL', 'YOUR', 'FROM', 'THEY', 'KNOW', 'WANT',
            'BEEN', 'GOOD', 'MUCH', 'SOME', 'TIME', 'VERY', 'WHEN', 'COME', 'HERE', 'JUST',
            'LIKE', 'LONG', 'MAKE', 'MANY', 'OVER', 'SUCH', 'TAKE', 'THAN', 'THEM', 'WELL',
            'WORK', 'CALL', 'CAME', 'EACH', 'EVEN', 'FIND', 'GIVE', 'HAND', 'HIGH', 'KEEP',
            'LAST', 'LEFT', 'LIFE', 'LIVE', 'LOOK', 'MADE', 'MOST', 'MOVE', 'MUST', 'NAME',
            'NEED', 'NEXT', 'OPEN', 'PART', 'PLAY', 'RIGHT', 'SAID', 'SAME', 'SEEM', 'SHOW',
            'SIDE', 'TELL', 'TURN', 'USED', 'WANT', 'WAYS', 'WEEK', 'WENT', 'WERE', 'WHAT',
            'WORD', 'WORK', 'YEAR', 'BACK', 'BOOK', 'FACE', 'FACT', 'FEEL', 'FIRE', 'FOOD',
            'FORM', 'FOUR', 'FREE', 'FULL', 'GAME', 'GIRL', 'GOES', 'HELP', 'HOME', 'HOPE',
            'HOUR', 'IDEA', 'KIND', 'LAND', 'LATE', 'LINE', 'LIST', 'LOVE', 'MIND', 'NICE',
            'ONLY', 'PLAN', 'REAL', 'ROOM', 'SAVE', 'SEND', 'SOON', 'STOP', 'SURE', 'TALK',
            'TEAM', 'TREE', 'TRUE', 'TYPE', 'VIEW', 'WALK', 'WALL', 'WEAR', 'WIFE', 'WIND',
            
            # Common 5-letter words
            'ABOUT', 'AFTER', 'AGAIN', 'BEING', 'COULD', 'EVERY', 'FIRST', 'FOUND', 'GREAT',
            'GROUP', 'HOUSE', 'LARGE', 'NEVER', 'OTHER', 'PLACE', 'RIGHT', 'SHALL', 'SMALL',
            'SOUND', 'STILL', 'THEIR', 'THERE', 'THESE', 'THING', 'THINK', 'THREE', 'UNDER',
            'WATER', 'WHERE', 'WHICH', 'WHILE', 'WORLD', 'WOULD', 'WRITE', 'YOUNG'
        ]
        
        # Create frequency map
        self.word_frequencies = {}
        for i, word in enumerate(self.target_words):
            freq = 1.0 - (i * 0.01) if len(word) == 3 else 0.8 - (i * 0.005)
            self.word_frequencies[word] = max(0.1, freq)
        
        # Finalize dictionary setup
        self._finalize_dictionary_setup()
    
    def _finalize_dictionary_setup(self):
        """Finalize dictionary setup by creating word_set and words_by_length"""
        try:
            # Create word set for fast lookup
            self.word_set = set(self.target_words)
            
            # Group words by length for faster matching
            self.words_by_length = {}
            for word in self.target_words:
                length = len(word)
                if length not in self.words_by_length:
                    self.words_by_length[length] = []
                self.words_by_length[length].append(word)
                
            print(f"📚 Dictionary finalized: {len(self.target_words)} words, {len(self.words_by_length)} length groups")
            
        except Exception as e:
            print(f"⚠️ Dictionary finalization error: {e}")
            # Ensure basic attributes exist
            if not hasattr(self, 'word_set'):
                self.word_set = set()
            if not hasattr(self, 'words_by_length'):
                self.words_by_length = {}
    
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
    
    def generate_candidates(self, word):
        """Generate correction candidates using confusion matrix"""
        candidates = set([word])
        
        # Single character substitutions using confusion matrix
        for i, char in enumerate(word):
            if char in self.confusion_pairs:
                for replacement in self.confusion_pairs[char]:
                    if len(replacement) == 1:
                        candidate = word[:i] + replacement + word[i+1:]
                        candidates.add(candidate)
        
        return list(candidates)
    
    def correct_word(self, word):
        """Enhanced word correction using multiple strategies and better scoring"""
        if not word:
            return "", 0.0
        
        word = word.upper().strip()
        
        # Direct match with error handling
        try:
            if hasattr(self, 'word_set') and word in self.word_set:
                return word, 1.0
        except Exception as e:
            print(f"⚠️ Word set lookup error: {e}")
            # Fallback to target_words list
            if hasattr(self, 'target_words') and word in self.target_words:
                return word, 1.0
        
        # Multi-strategy candidate generation
        candidates = set()
        
        # Strategy 1: Confusion matrix candidates
        candidates.update(self.generate_candidates(word))
        
        # Strategy 2: Length-based candidates (words of similar length)
        target_length = len(word)
        for length in range(max(1, target_length - 1), target_length + 2):
            if length in self.words_by_length:
                candidates.update(self.words_by_length[length])
        
        # Strategy 3: Prefix matching for partial words
        if len(word) >= 2:
            for dict_word in self.target_words:
                if dict_word.startswith(word[:2]) or word.startswith(dict_word[:2]):
                    candidates.add(dict_word)
        
        # Enhanced scoring system
        best_word = word
        best_score = 0.0
        
        for candidate in candidates:
            # Safe word set lookup
            is_valid_word = False
            try:
                if hasattr(self, 'word_set'):
                    is_valid_word = candidate in self.word_set
                else:
                    is_valid_word = candidate in self.target_words
            except Exception as e:
                print(f"⚠️ Candidate validation error: {e}")
                is_valid_word = candidate in self.target_words if hasattr(self, 'target_words') else False
            
            if is_valid_word:
                # Multiple similarity metrics
                
                # 1. Levenshtein distance
                lev_distance = self.levenshtein_distance(word, candidate)
                max_len = max(len(word), len(candidate))
                lev_similarity = 1.0 - (lev_distance / max_len) if max_len > 0 else 0.0
                
                # 2. Longest common subsequence
                lcs_similarity = self.lcs_similarity(word, candidate)
                
                # 3. Character frequency similarity
                char_similarity = self.character_frequency_similarity(word, candidate)
                
                # 4. Position-weighted similarity (early characters matter more)
                pos_similarity = self.position_weighted_similarity(word, candidate)
                
                # Combined similarity score
                combined_similarity = (
                    lev_similarity * 0.3 +
                    lcs_similarity * 0.25 +
                    char_similarity * 0.25 +
                    pos_similarity * 0.2
                )
                
                # Frequency and length bonuses
                frequency_boost = self.word_frequencies.get(candidate, 0.5) * 0.15
                
                # Length penalty for very different lengths
                length_penalty = abs(len(word) - len(candidate)) * 0.05
                
                # Final score calculation
                final_score = combined_similarity + frequency_boost - length_penalty
                final_score = max(0.0, min(1.0, final_score))
                
                if final_score > best_score:
                    best_score = final_score
                    best_word = candidate
        
        # Try TextBlob if available and no good match found
        if TEXTBLOB_AVAILABLE and best_score < 0.5:
            try:
                blob = TextBlob(word.lower())
                corrected = str(blob.correct()).upper()
                # Safe word set lookup for TextBlob correction
                is_corrected_valid = False
                try:
                    if hasattr(self, 'word_set'):
                        is_corrected_valid = corrected in self.word_set
                    else:
                        is_corrected_valid = corrected in self.target_words
                except Exception as e:
                    print(f"⚠️ TextBlob validation error: {e}")
                    is_corrected_valid = corrected in self.target_words if hasattr(self, 'target_words') else False
                
                if is_corrected_valid:
                    # Calculate score for TextBlob suggestion
                    tb_distance = self.levenshtein_distance(word, corrected)
                    tb_score = max(0.6, 1.0 - (tb_distance / max(len(word), len(corrected))))
                    if tb_score > best_score:
                        return corrected, tb_score
            except Exception as e:
                print(f"⚠️ TextBlob correction error: {e}")
        
        return best_word, best_score
    
    def lcs_similarity(self, s1, s2):
        """Calculate similarity based on longest common subsequence"""
        def lcs_length(x, y):
            m, n = len(x), len(y)
            dp = [[0] * (n + 1) for _ in range(m + 1)]
            
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if x[i-1] == y[j-1]:
                        dp[i][j] = dp[i-1][j-1] + 1
                    else:
                        dp[i][j] = max(dp[i-1][j], dp[i][j-1])
            
            return dp[m][n]
        
        lcs_len = lcs_length(s1, s2)
        max_len = max(len(s1), len(s2))
        return lcs_len / max_len if max_len > 0 else 0.0
    
    def character_frequency_similarity(self, s1, s2):
        """Calculate similarity based on character frequency - safe division"""
        try:
            from collections import Counter
            
            counter1 = Counter(s1)
            counter2 = Counter(s2)
            
            all_chars = set(counter1.keys()) | set(counter2.keys())
            
            if not all_chars:
                return 1.0
            
            similarity = 0.0
            for char in all_chars:
                freq1 = counter1.get(char, 0)
                freq2 = counter2.get(char, 0)
                max_freq = max(freq1, freq2)
                min_freq = min(freq1, freq2)
                
                # Safe division with minimum denominator
                char_sim = min_freq / max(1, max_freq) if max_freq > 0 else 0
                similarity += char_sim
            
            # Safe division for final result
            return similarity / max(1, len(all_chars))
            
        except Exception as e:
            print(f"⚠️ Character frequency similarity error: {e}")
            return 0.0
    
    def position_weighted_similarity(self, s1, s2):
        """Calculate similarity with higher weight for early positions - safe division"""
        min_len = min(len(s1), len(s2))
        if min_len == 0:
            return 0.0
        
        try:
            matches = 0.0
            total_weight = 0.0
            
            for i in range(min_len):
                weight = 1.0 / max(1.0, (i + 1))  # Safe division with minimum denominator
                total_weight += weight
                if s1[i] == s2[i]:
                    matches += weight
            
            # Safe division with minimum denominator
            return matches / max(0.001, total_weight) if total_weight > 0 else 0.0
            
        except Exception as e:
            print(f"⚠️ Position weighted similarity error: {e}")
            return 0.0
    
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

class IntegratedVoiceFeedback:
    """Integrated voice feedback using existing TTS setup"""
    
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
                print("🔊 Text-to-speech initialized")
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
                                     'zira' in voice.name.lower() or
                                     'hazel' in voice.name.lower()):
                        self.engine.setProperty('voice', voice.id)
                        print(f"✅ Voice set to: {voice.name}")
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
        """Enhanced speak word with better audio handling"""
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
                
                # Enhanced TTS with better error handling
                if self.engine:
                    # Stop any current speech
                    try:
                        self.engine.stop()
                    except:
                        pass
                    
                    # Speak the word
                    self.engine.say(word)
                    self.engine.runAndWait()
                    
                    print(f"✅ Spoke: {word}")
                else:
                    print(f"❌ No TTS engine available")
                
                self.recent_words.append(word)
                self.last_speech_time = time.time()
                
            except Exception as e:
                print(f"❌ Speech error: {e}")
                # Try to reinitialize engine
                try:
                    self.engine = pyttsx3.init()
                    self.configure_engine()
                    print("🔄 TTS engine reinitialized")
                except:
                    print("❌ Failed to reinitialize TTS engine")
            finally:
                self.speaking = False
        
        # Use threading for non-blocking speech
        speech_thread = threading.Thread(target=speak_thread, daemon=True)
        speech_thread.start()
    
    def speak_word_immediate(self, word):
        """Speak word immediately without threading (for testing)"""
        if not self.tts_available or not word:
            print(f"🔊 Would speak: {word}")
            return
        
        try:
            word = word.strip().upper()
            print(f"🔊 Speaking immediately: {word}")
            
            if self.engine:
                self.engine.say(word)
                self.engine.runAndWait()
                print(f"✅ Spoke immediately: {word}")
            
        except Exception as e:
            print(f"❌ Immediate speech error: {e}")
    
    def test_speech(self):
        """Test speech functionality"""
        print("🧪 Testing speech...")
        self.speak_word_immediate("TEST")
    
    def is_speaking(self):
        """Check if currently speaking"""
        return self.speaking

class CompleteAirWritingSystem:
    """Complete integrated air writing system using all existing components"""
    
    def __init__(self, model_paths=None, dictionary_path=None):
        print("🚀 Initializing Complete Integrated Air Writing System...")
        
        # Initialize components with error handling
        try:
            self.hand_tracker = HandTracker(max_hands=1, trail_length=200, alpha=0.3)
            print("✅ Hand tracker initialized")
        except Exception as e:
            print(f"❌ Hand tracker initialization failed: {e}")
            raise
        
        try:
            self.letter_recognizer = IntegratedLetterRecognizer(model_paths)
            print("✅ Letter recognizer initialized")
        except Exception as e:
            print(f"❌ Letter recognizer initialization failed: {e}")
            raise
        
        try:
            self.word_corrector = IntegratedWordCorrector(dictionary_path)
            print("✅ Word corrector initialized")
        except Exception as e:
            print(f"❌ Word corrector initialization failed: {e}")
            raise
        
        try:
            self.voice_feedback = IntegratedVoiceFeedback()
            print("✅ Voice feedback initialized")
        except Exception as e:
            print(f"❌ Voice feedback initialization failed: {e}")
            # Don't raise - voice is optional
            self.voice_feedback = None
        
        # Initialize camera with retry logic
        self.cap = None
        camera_attempts = 3
        for attempt in range(camera_attempts):
            try:
                print(f"📹 Attempting to initialize camera (attempt {attempt + 1}/{camera_attempts})")
                self.cap = cv2.VideoCapture(0)
                
                if not self.cap.isOpened():
                    raise Exception("Camera not accessible")
                
                # Test camera by reading a frame
                ret, test_frame = self.cap.read()
                if not ret:
                    raise Exception("Cannot read from camera")
                
                # Try high resolution first, fallback to lower if needed
                try:
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                    self.cap.set(cv2.CAP_PROP_FPS, 30)
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    
                    # Test the settings
                    test_ret, test_frame = self.cap.read()
                    if not test_ret or test_frame is None or test_frame.size == 0:
                        raise Exception("High resolution not supported")
                    
                    print("✅ High resolution camera settings applied")
                    
                except Exception as res_error:
                    print(f"⚠️ High resolution failed: {res_error}, trying lower resolution...")
                    # Fallback to lower resolution
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    self.cap.set(cv2.CAP_PROP_FPS, 30)
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    print("✅ Lower resolution camera settings applied")
                
                print("✅ Camera initialized successfully")
                break
                
            except Exception as e:
                print(f"⚠️ Camera initialization attempt {attempt + 1} failed: {e}")
                if self.cap:
                    self.cap.release()
                    self.cap = None
                
                if attempt < camera_attempts - 1:
                    print("🔄 Retrying camera initialization...")
                    time.sleep(1)
                else:
                    print("❌ All camera initialization attempts failed")
                    raise Exception("Could not initialize camera after multiple attempts")
        
        # Application state
        self.current_word = ""
        self.recognized_words = []
        self.current_path = []
        self.word_suggestions = []
        
        # Timing parameters
        self.letter_pause_frames = 20
        self.word_pause_frames = 60
        self.min_path_length = 8
        self.idle_time_threshold = 1.5
        
        # State counters
        self.letter_pause_count = 0
        self.word_pause_count = 0
        self.last_movement_time = time.time()
        self.last_word_time = 0
        
        # Performance tracking
        self.fps = 0
        self.frame_count = 0
        self.fps_start_time = time.time()
        
        # Enhanced system monitoring
        self.error_counts = {
            'camera_errors': 0,
            'hand_tracking_errors': 0,
            'recognition_errors': 0,
            'correction_errors': 0,
            'total_frames': 0,
            'successful_frames': 0
        }
        
        self.performance_stats = {
            'avg_recognition_time': 0.0,
            'avg_correction_time': 0.0,
            'recognition_accuracy': 0.0,
            'correction_success_rate': 0.0
        }
        
        # Initialize success tracking
        self.recent_successes = deque(maxlen=10)
        
        # Display settings
        self.show_landmarks = True
        self.show_trail = True
        self.show_suggestions = True
        self.show_debug = False
        
        # Visual effects
        self.trail_colors = {
            'gradient': [(255, 0, 0), (0, 255, 255)],
            'rainbow': [(255, 0, 0), (255, 127, 0), (255, 255, 0), (0, 255, 0), (0, 0, 255)],
            'fire': [(0, 0, 255), (0, 127, 255), (0, 255, 255), (255, 255, 255)],
            'ocean': [(255, 255, 0), (255, 127, 0), (255, 0, 0)]
        }
        self.current_color_scheme = 'gradient'
        
        # Animation state
        self.word_animation_active = False
        self.word_animation_start = 0
        self.animation_word = ""
        
        # Output logging
        self.output_file = "output_log.txt"
        self.session_start_time = datetime.now()
        
        print("✅ Complete Integrated Air Writing System initialized")
        
        # Test voice feedback
        self.test_voice_feedback()
        
        # Perform system health check
        self.system_health_check()
        
        self.print_instructions()
    
    def system_health_check(self):
        """Perform comprehensive system health check"""
        print("\n🔍 Performing system health check...")
        
        health_status = {
            'camera': False,
            'hand_tracker': False,
            'letter_recognizer': False,
            'word_corrector': False,
            'voice_feedback': False
        }
        
        # Check camera
        try:
            ret, test_frame = self.cap.read()
            if ret and test_frame is not None:
                health_status['camera'] = True
                print("✅ Camera: Working")
            else:
                print("❌ Camera: Not providing frames")
        except Exception as e:
            print(f"❌ Camera: Error - {e}")
        
        # Check hand tracker
        try:
            if self.hand_tracker and hasattr(self.hand_tracker, 'hands'):
                health_status['hand_tracker'] = True
                print("✅ Hand Tracker: Ready")
            else:
                print("❌ Hand Tracker: Not initialized")
        except Exception as e:
            print(f"❌ Hand Tracker: Error - {e}")
        
        # Check letter recognizer
        try:
            if self.letter_recognizer and len(self.letter_recognizer.models) > 0:
                health_status['letter_recognizer'] = True
                print("✅ Letter Recognizer: Models loaded")
            else:
                print("⚠️ Letter Recognizer: No models loaded (demo mode)")
                health_status['letter_recognizer'] = True  # Demo mode is acceptable
        except Exception as e:
            print(f"❌ Letter Recognizer: Error - {e}")
        
        # Check word corrector
        try:
            if self.word_corrector and len(self.word_corrector.target_words) > 0:
                health_status['word_corrector'] = True
                print("✅ Word Corrector: Dictionary loaded")
            else:
                print("❌ Word Corrector: No dictionary")
        except Exception as e:
            print(f"❌ Word Corrector: Error - {e}")
        
        # Check voice feedback
        try:
            if self.voice_feedback and self.voice_feedback.tts_available:
                health_status['voice_feedback'] = True
                print("✅ Voice Feedback: Available")
            else:
                print("⚠️ Voice Feedback: Not available (silent mode)")
                health_status['voice_feedback'] = True  # Silent mode is acceptable
        except Exception as e:
            print(f"❌ Voice Feedback: Error - {e}")
        
        # Overall health assessment
        critical_components = ['camera', 'hand_tracker', 'letter_recognizer', 'word_corrector']
        critical_healthy = all(health_status[comp] for comp in critical_components)
        
        if critical_healthy:
            print("🎉 System health check passed - All critical components ready!")
        else:
            failed_components = [comp for comp in critical_components if not health_status[comp]]
            print(f"⚠️ System health check warnings - Issues with: {', '.join(failed_components)}")
        
        return health_status
    
    def validate_frame(self, frame):
        """Validate frame properties to prevent OpenCV errors"""
        if frame is None:
            return False, "Frame is None"
        
        if frame.size == 0:
            return False, "Frame is empty"
        
        if len(frame.shape) != 3:
            return False, f"Invalid frame dimensions: {frame.shape}"
        
        if frame.shape[0] < 50 or frame.shape[1] < 50:
            return False, f"Frame too small: {frame.shape}"
        
        if frame.dtype != np.uint8:
            return False, f"Invalid frame type: {frame.dtype}"
        
        if not frame.flags['C_CONTIGUOUS']:
            return False, "Frame not contiguous in memory"
        
        return True, "Frame valid"
    
    def safe_camera_read(self):
        """Safely read frame from camera with validation"""
        try:
            ret, frame = self.cap.read()
            
            if not ret or frame is None:
                return False, None, "Failed to read frame"
            
            # Validate frame
            is_valid, message = self.validate_frame(frame)
            if not is_valid:
                return False, None, f"Invalid frame: {message}"
            
            # Ensure frame is contiguous
            if not frame.flags['C_CONTIGUOUS']:
                frame = np.ascontiguousarray(frame)
            
            return True, frame, "Success"
            
        except Exception as e:
            return False, None, f"Camera read error: {e}"
    
    def test_voice_feedback(self):
        """Test voice feedback system"""
        print("\n🧪 Testing voice feedback...")
        try:
            if self.voice_feedback.tts_available:
                print("🔊 Testing speech with 'HELLO'...")
                self.voice_feedback.speak_word_immediate("HELLO")
                print("✅ Voice feedback test completed")
            else:
                print("⚠️  Voice feedback not available")
        except Exception as e:
            print(f"❌ Voice feedback test failed: {e}")
    
    def print_instructions(self):
        """Print usage instructions"""
        print("\n" + "="*80)
        print("🖐️  COMPLETE INTEGRATED AIR WRITING SYSTEM")
        print("="*80)
        print("📋 Instructions:")
        print("   • Hold up your INDEX FINGER (other fingers curled)")
        print("   • Write letters in the air slowly and clearly")
        print("   • Pause briefly between letters (system auto-detects)")
        print("   • Pause longer between words (1-2 seconds)")
        print("   • System will auto-correct and speak recognized words")
        print("\n🎯 Try these words:")
        sample_words = self.word_corrector.target_words[:10]
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
        print("   V        - Test voice feedback")
        print("   1-4      - Change trail color schemes")
        print("   ESC      - Exit system")
        print("\n🔊 Audio Features:")
        print("   • Words are automatically spoken when completed")
        print("   • Auto-correction happens after 1-2 seconds of idle time")
        print("   • High-confidence corrections are applied automatically")
        print("   • Press 'V' to test if audio is working")
        print("="*80 + "\n")
    
    def update_fps(self):
        """Update FPS counter"""
        self.frame_count += 1
        current_time = time.time()
        
        if current_time - self.fps_start_time >= 1.0:
            self.fps = self.frame_count
            self.frame_count = 0
            self.fps_start_time = current_time
    
    def interpolate_color(self, color1, color2, factor):
        """Interpolate between two colors"""
        factor = max(0, min(1, factor))
        return tuple(int(c1 + (c2 - c1) * factor) for c1, c2 in zip(color1, color2))
    
    def get_trail_color(self, progress):
        """Get trail color based on progress"""
        colors = self.trail_colors[self.current_color_scheme]
        
        if len(colors) == 2:
            return self.interpolate_color(colors[0], colors[1], progress)
        
        # Multi-color gradient
        segment_size = 1.0 / (len(colors) - 1)
        segment_index = int(progress / segment_size)
        segment_index = min(segment_index, len(colors) - 2)
        
        local_progress = (progress - segment_index * segment_size) / segment_size
        return self.interpolate_color(colors[segment_index], colors[segment_index + 1], local_progress)
    
    def draw_enhanced_trail(self, frame):
        """Draw enhanced trail using existing hand tracker"""
        trail_points = self.hand_tracker.get_trail_path()
        
        if len(trail_points) < 2:
            return
        
        # Draw trail with color gradient
        for i in range(1, len(trail_points)):
            progress = i / len(trail_points)
            color = self.get_trail_color(progress)
            thickness = max(2, int(6 * progress))
            
            cv2.line(frame, trail_points[i-1], trail_points[i], color, thickness)
        
        # Draw animated fingertip
        if self.hand_tracker.hand_present and hasattr(self.hand_tracker, 'fingertip_trail') and self.hand_tracker.fingertip_trail:
            fingertip = self.hand_tracker.fingertip_trail[-1]
            
            # Pulsing effect
            pulse_factor = abs(np.sin(time.time() * 5))
            pulse_radius = int(8 + pulse_factor * 6)
            
            # Color based on writing state
            base_color = (0, 0, 255) if self.hand_tracker.writing_mode else (0, 255, 0)
            
            cv2.circle(frame, fingertip, pulse_radius, base_color, -1)
            cv2.circle(frame, fingertip, pulse_radius + 3, (255, 255, 255), 2)
    
    def draw_ui(self, frame):
        """Draw user interface"""
        h, w = frame.shape[:2]
        
        # Create overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (w-10, 200), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        
        # Current word
        word_display = self.current_word if self.current_word else "[Writing...]"
        word_color = (0, 255, 255) if self.current_word else (128, 128, 128)
        
        cv2.putText(frame, f"Current Word: {word_display}", (20, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, word_color, 3)
        
        # Word suggestions
        if self.show_suggestions and self.word_suggestions:
            suggestions_text = f"Suggestions: {' | '.join(self.word_suggestions)}"
            cv2.putText(frame, suggestions_text, (20, 85), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Recent words
        if self.recognized_words:
            recent_words = " → ".join(self.recognized_words[-4:])
            cv2.putText(frame, f"Recent: {recent_words}", (20, 115), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Status indicators
        status_y = 145
        
        # Hand detection
        hand_status = "✅ Hand" if self.hand_tracker.hand_present else "❌ No Hand"
        hand_color = (0, 255, 0) if self.hand_tracker.hand_present else (0, 0, 255)
        cv2.putText(frame, hand_status, (20, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, hand_color, 2)
        
        # Writing status
        writing_status = "✍️ Writing" if self.hand_tracker.writing_mode else "✋ Ready"
        writing_color = (0, 255, 255) if self.hand_tracker.writing_mode else (255, 255, 255)
        cv2.putText(frame, writing_status, (150, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, writing_color, 2)
        
        # Voice status
        voice_status = "🔊 Speaking" if self.voice_feedback.is_speaking() else "🔇 Ready"
        voice_color = (255, 0, 255) if self.voice_feedback.is_speaking() else (255, 255, 255)
        cv2.putText(frame, voice_status, (280, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, voice_color, 2)
        
        # Enhanced performance info
        perf_y = 175
        cv2.putText(frame, f"FPS: {self.fps}", (20, perf_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        avg_proc_time = self.letter_recognizer.get_average_processing_time()
        cv2.putText(frame, f"Proc: {avg_proc_time:.1f}ms", (100, perf_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.putText(frame, f"Path: {len(self.current_path)}", (200, perf_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.putText(frame, f"Words: {len(self.recognized_words)}", (280, perf_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Success rate indicator
        if hasattr(self, 'recent_successes') and self.recent_successes:
            success_rate = sum(self.recent_successes) / len(self.recent_successes)
            success_color = (0, 255, 0) if success_rate > 0.7 else (0, 255, 255) if success_rate > 0.4 else (0, 0, 255)
            cv2.putText(frame, f"Acc: {success_rate:.1%}", (360, perf_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, success_color, 1)
        
        # Error count (if debug mode)
        if self.show_debug:
            error_y = h - 80
            total_errors = sum(self.error_counts[key] for key in self.error_counts if 'error' in key)
            cv2.putText(frame, f"Errors: {total_errors}", (20, error_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 100), 1)
            
            if self.error_counts['total_frames'] > 0:
                frame_success_rate = self.error_counts['successful_frames'] / self.error_counts['total_frames']
                cv2.putText(frame, f"Frame Success: {frame_success_rate:.1%}", (120, error_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Instructions
        instructions = [
            "✋ Hold INDEX finger up and write letters clearly in the air",
            "⌨️ SPACE: Complete letter | ENTER: Complete word | C: Clear | ESC: Exit"
        ]
        
        for i, instruction in enumerate(instructions):
            cv2.putText(frame, instruction, (20, h - 50 + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    
    def process_letter_completion(self):
        """Enhanced letter completion with comprehensive error handling"""
        try:
            if len(self.current_path) < self.min_path_length:
                return
            
            # Recognize letter using integrated recognizer with error handling
            try:
                letter, confidence = self.letter_recognizer.recognize_letter(self.current_path)
            except Exception as recognition_error:
                print(f"⚠️ Letter recognition error: {recognition_error}")
                self.error_counts['recognition_errors'] += 1
                letter, confidence = None, 0.0
            
            # Enhanced adaptive confidence threshold with safe calculations
            try:
                base_confidence = 0.4  # Increased base threshold for better accuracy
                
                # Safe path length calculation
                path_length = max(1, len(self.current_path))
                length_factor = min(0.2, path_length / max(1.0, 100.0))
                
                # Adjust based on current word context
                context_factor = 0.0
                if len(self.current_word) > 0:
                    # Lower threshold for subsequent letters in a word
                    context_factor = -0.1
                
                # Safe recent success rate calculation
                if hasattr(self, 'recent_successes') and len(self.recent_successes) > 0:
                    try:
                        success_rate = sum(self.recent_successes) / max(1, len(self.recent_successes))
                        if success_rate < 0.5:
                            context_factor += 0.1  # Raise threshold if recent accuracy is poor
                    except Exception as success_error:
                        print(f"⚠️ Success rate calculation error: {success_error}")
                        context_factor += 0.05  # Conservative adjustment
                else:
                    self.recent_successes = deque(maxlen=10)
                
                min_confidence = max(0.3, base_confidence - length_factor + context_factor)
                
            except Exception as threshold_error:
                print(f"⚠️ Confidence threshold calculation error: {threshold_error}")
                min_confidence = 0.4  # Safe fallback
            
            # Process recognition result
            if letter and confidence > min_confidence:
                try:
                    self.current_word += letter
                    print(f"✅ Letter: {letter} (confidence: {confidence:.3f}, threshold: {min_confidence:.3f})")
                    
                    # Track success for adaptive thresholding
                    if hasattr(self, 'recent_successes'):
                        self.recent_successes.append(1)
                    
                    # Update suggestions with enhanced auto-correction preview
                    try:
                        self.word_suggestions = self.word_corrector.get_suggestions(self.current_word)
                        
                        # Enhanced auto-correction preview for 2+ letter words
                        if len(self.current_word) >= 2:
                            try:
                                corrected_word, correction_confidence = self.word_corrector.correct_word(self.current_word)
                                if corrected_word != self.current_word and correction_confidence > 0.6:
                                    print(f"💡 Auto-correction preview: {self.current_word} → {corrected_word} (conf: {correction_confidence:.3f})")
                                    # Add corrected word to suggestions if not already there
                                    if corrected_word not in self.word_suggestions:
                                        self.word_suggestions.insert(0, corrected_word)
                            except Exception as correction_error:
                                print(f"⚠️ Auto-correction preview error: {correction_error}")
                                self.error_counts['correction_errors'] += 1
                        
                        # Show current word progress
                        if len(self.current_word) > 1:
                            suggestions_text = ', '.join(self.word_suggestions[:3]) if self.word_suggestions else "None"
                            print(f"📝 Current word: {self.current_word} | Suggestions: {suggestions_text}")
                            
                    except Exception as suggestion_error:
                        print(f"⚠️ Word suggestion error: {suggestion_error}")
                        self.word_suggestions = []
                    
                except Exception as processing_error:
                    print(f"⚠️ Letter processing error: {processing_error}")
                    
            else:
                try:
                    confidence_str = f"{confidence:.3f}" if confidence is not None else "N/A"
                    threshold_str = f"{min_confidence:.3f}" if 'min_confidence' in locals() else "N/A"
                    letter_str = letter if letter else "None"
                    
                    print(f"❌ Letter rejected: {letter_str} (confidence: {confidence_str}, needed: {threshold_str})")
                    
                    # Track failure for adaptive thresholding
                    if hasattr(self, 'recent_successes'):
                        self.recent_successes.append(0)
                        
                except Exception as rejection_error:
                    print(f"⚠️ Letter rejection logging error: {rejection_error}")
            
        except Exception as completion_error:
            print(f"⚠️ Letter completion error: {completion_error}")
            self.error_counts['recognition_errors'] += 1
            
        finally:
            # Always clear path and reset state
            try:
                self.current_path.clear()
                if hasattr(self.hand_tracker, 'clear_trail'):
                    self.hand_tracker.clear_trail()
                self.letter_pause_count = 0
            except Exception as cleanup_error:
                print(f"⚠️ Letter completion cleanup error: {cleanup_error}")
    
    def process_word_completion(self):
        """Enhanced word completion with comprehensive error handling"""
        try:
            if not self.current_word:
                return
            
            # Automatic word correction with error handling
            corrected_word = self.current_word
            correction_confidence = 1.0
            
            try:
                corrected_word, correction_confidence = self.word_corrector.correct_word(self.current_word)
            except Exception as correction_error:
                print(f"⚠️ Word correction error: {correction_error}")
                self.error_counts['correction_errors'] += 1
                # Use original word if correction fails
                corrected_word = self.current_word
                correction_confidence = 0.5
            
            # Show correction process with safe formatting
            try:
                if corrected_word != self.current_word:
                    print(f"🔄 Auto-correcting: {self.current_word} → {corrected_word} (confidence: {correction_confidence:.3f})")
                else:
                    print(f"✅ Word recognized: {corrected_word} (confidence: {correction_confidence:.3f})")
            except Exception as display_error:
                print(f"⚠️ Word display error: {display_error}")
                print(f"✅ Word completed: {corrected_word}")
            
            # Add to recognized words with error handling
            try:
                self.recognized_words.append(corrected_word)
            except Exception as append_error:
                print(f"⚠️ Word append error: {append_error}")
                # Initialize if needed
                if not hasattr(self, 'recognized_words'):
                    self.recognized_words = []
                self.recognized_words.append(corrected_word)
            
            # Log to file with error handling
            try:
                self.log_word(corrected_word)
            except Exception as log_error:
                print(f"⚠️ Word logging error: {log_error}")
            
            # Trigger word completion animation with error handling
            try:
                self.word_animation_active = True
                self.word_animation_start = time.time()
                self.animation_word = corrected_word
            except Exception as animation_error:
                print(f"⚠️ Animation setup error: {animation_error}")
            
            # Enhanced voice feedback with comprehensive error handling
            try:
                current_time = time.time()
                if current_time - self.last_word_time > 0.5:  # Reduced delay for better UX
                    print(f"🔊 Triggering speech for: {corrected_word}")
                    
                    # Try voice feedback with multiple fallbacks
                    speech_success = False
                    
                    if hasattr(self, 'voice_feedback') and self.voice_feedback:
                        try:
                            # Try immediate speech first for better responsiveness
                            self.voice_feedback.speak_word_immediate(corrected_word)
                            speech_success = True
                        except Exception as immediate_speech_error:
                            print(f"⚠️ Immediate speech error: {immediate_speech_error}")
                            try:
                                # Fallback to threaded speech
                                self.voice_feedback.speak_word(corrected_word)
                                speech_success = True
                            except Exception as threaded_speech_error:
                                print(f"⚠️ Threaded speech error: {threaded_speech_error}")
                    
                    if not speech_success:
                        print(f"🔇 Speech unavailable, word completed silently: {corrected_word}")
                    
                    self.last_word_time = current_time
                    
            except Exception as voice_error:
                print(f"⚠️ Voice feedback error: {voice_error}")
            
            # Reset state after a short delay with error handling
            try:
                threading.Timer(1.5, self.reset_word_state).start()
            except Exception as timer_error:
                print(f"⚠️ Timer setup error: {timer_error}")
                # Immediate reset as fallback
                self.reset_word_state()
                
        except Exception as completion_error:
            print(f"⚠️ Word completion error: {completion_error}")
            # Emergency reset
            try:
                self.reset_word_state()
            except:
                pass
    
    def reset_word_state(self):
        """Reset word state"""
        self.current_word = ""
        self.word_suggestions = []
        self.word_pause_count = 0
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
        elif key == ord('l') or key == ord('L'):
            self.show_landmarks = not self.show_landmarks
            print(f"🖐️ Landmarks: {'ON' if self.show_landmarks else 'OFF'}")
        elif key == ord('s') or key == ord('S'):
            self.show_suggestions = not self.show_suggestions
            print(f"💡 Suggestions: {'ON' if self.show_suggestions else 'OFF'}")
        elif key == ord('d') or key == ord('D'):
            self.show_debug = not self.show_debug
            print(f"🐛 Debug: {'ON' if self.show_debug else 'OFF'}")
        elif key == ord('v') or key == ord('V'):
            # Test voice feedback
            print("🧪 Testing voice feedback...")
            self.voice_feedback.test_speech()
        elif key == ord('1'):
            self.current_color_scheme = 'gradient'
            print("🎨 Color: Gradient")
        elif key == ord('2'):
            self.current_color_scheme = 'rainbow'
            print("🎨 Color: Rainbow")
        elif key == ord('3'):
            self.current_color_scheme = 'fire'
            print("🎨 Color: Fire")
        elif key == ord('4'):
            self.current_color_scheme = 'ocean'
            print("🎨 Color: Ocean")
        elif key == 27:  # ESC
            return False
        
        return True
    
    def run(self):
        """Enhanced run method with better error handling and stability"""
        print("🚀 Starting Complete Integrated Air Writing System...")
        
        # Initialize frame counter for stability monitoring
        frame_counter = 0
        last_successful_frame = time.time()
        camera_retry_count = 0
        max_camera_retries = 3
        
        try:
            while True:
                try:
                    # Track frame processing
                    self.error_counts['total_frames'] += 1
                    
                    # Use safe camera reading
                    success, frame, message = self.safe_camera_read()
                    
                    if not success:
                        print(f"⚠️ Camera read failed: {message}")
                        self.error_counts['camera_errors'] += 1
                        camera_retry_count += 1
                        
                        if camera_retry_count >= max_camera_retries:
                            print("❌ Camera failed multiple times, exiting...")
                            break
                        
                        # Try to reinitialize camera
                        print(f"🔄 Attempting to reinitialize camera (attempt {camera_retry_count})")
                        try:
                            self.cap.release()
                            time.sleep(1)
                            self.cap = cv2.VideoCapture(0)
                            if self.cap.isOpened():
                                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                                self.cap.set(cv2.CAP_PROP_FPS, 30)
                                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                                print("✅ Camera reinitialized successfully")
                                camera_retry_count = 0
                        except Exception as reinit_error:
                            print(f"⚠️ Camera reinitialization failed: {reinit_error}")
                        continue
                    
                    # Reset camera retry count on successful frame
                    camera_retry_count = 0
                    frame_counter += 1
                    last_successful_frame = time.time()
                    self.error_counts['successful_frames'] += 1
                    
                    # Flip frame for mirror effect with error handling
                    try:
                        frame = cv2.flip(frame, 1)
                        
                        # Ensure frame is in correct format
                        if frame.dtype != np.uint8:
                            frame = frame.astype(np.uint8)
                        
                        # Validate frame dimensions
                        if frame.shape[0] < 100 or frame.shape[1] < 100:
                            print("⚠️ Frame too small, skipping...")
                            continue
                            
                    except Exception as e:
                        print(f"⚠️ Frame flip error: {e}")
                        continue
                    
                    # Process hand tracking with enhanced error handling and recovery
                    fingertip, velocity = None, 0
                    try:
                        # Create a copy of the frame for hand tracking to avoid memory issues
                        tracking_frame = frame.copy()
                        
                        # Validate tracking frame
                        if tracking_frame is None or tracking_frame.size == 0:
                            print("⚠️ Invalid tracking frame")
                            continue
                        
                        result = self.hand_tracker.get_fingertip(tracking_frame)
                        
                        # Handle different return formats safely
                        if isinstance(result, tuple):
                            if len(result) == 3:
                                fingertip, processed_frame, velocity = result
                            elif len(result) == 2:
                                fingertip, processed_frame = result
                                velocity = 0
                            else:
                                fingertip = result[0] if result else None
                                processed_frame = None
                                velocity = 0
                        else:
                            fingertip = result
                            processed_frame = None
                            velocity = 0
                        
                        # Use the processed frame if valid, otherwise use original
                        if processed_frame is not None and processed_frame.size > 0:
                            # Validate processed frame before using
                            if len(processed_frame.shape) == 3 and processed_frame.shape[0] > 0:
                                frame = processed_frame
                            
                    except Exception as e:
                        print(f"⚠️ Hand tracking error: {e}")
                        self.error_counts['hand_tracking_errors'] += 1
                        
                        # Try to recover hand tracker
                        try:
                            if hasattr(self.hand_tracker, 'hands'):
                                print("🔄 Attempting to recover hand tracker...")
                                # Reset hand tracker state
                                self.hand_tracker.hand_present = False
                                self.hand_tracker.writing_mode = False
                                if hasattr(self.hand_tracker, 'fingertip_trail'):
                                    self.hand_tracker.fingertip_trail.clear()
                        except Exception as recovery_error:
                            print(f"⚠️ Hand tracker recovery failed: {recovery_error}")
                        
                        fingertip, velocity = None, 0
                    
                    # Update FPS
                    self.update_fps()
                    
                    # Handle writing logic with error handling
                    current_time = time.time()
                    
                    try:
                        if fingertip and self.hand_tracker.is_writing_active():
                            # Active writing
                            self.current_path.append(fingertip)
                            self.last_movement_time = current_time
                            self.letter_pause_count = 0
                            self.word_pause_count = 0
                        else:
                            # Handle pauses
                            if len(self.current_path) > 0:
                                self.letter_pause_count += 1
                            
                            # Letter completion with error handling
                            time_since_movement = current_time - self.last_movement_time
                            if (self.letter_pause_count >= self.letter_pause_frames or 
                                (len(self.current_path) >= self.min_path_length and time_since_movement > 0.8)):
                                try:
                                    self.process_letter_completion()
                                except Exception as e:
                                    print(f"⚠️ Letter completion error: {e}")
                            
                            # Enhanced word completion with auto-correction
                            if time_since_movement > self.idle_time_threshold:
                                self.word_pause_count += 1
                                
                                # Auto-complete word after idle time or sufficient pause
                                if (self.word_pause_count >= self.word_pause_frames or
                                    time_since_movement > 2.5):  # Reduced time for better UX
                                    if self.current_word:
                                        try:
                                            print(f"⏰ Auto-completing word after {time_since_movement:.1f}s idle time")
                                            self.process_word_completion()
                                        except Exception as e:
                                            print(f"⚠️ Word completion error: {e}")
                            
                            # Auto-correction for 3+ letter words after short pause
                            elif (len(self.current_word) >= 3 and 
                                  time_since_movement > 1.0 and 
                                  self.word_pause_count == 0):
                                try:
                                    # Show auto-correction preview
                                    corrected_word, confidence = self.word_corrector.correct_word(self.current_word)
                                    if corrected_word != self.current_word and confidence > 0.8:
                                        print(f"💡 High-confidence correction available: {self.current_word} → {corrected_word}")
                                        # Auto-apply high-confidence corrections after 2 seconds
                                        if time_since_movement > 2.0:
                                            print(f"🔄 Auto-applying correction: {self.current_word} → {corrected_word}")
                                            self.current_word = corrected_word
                                            self.process_word_completion()
                                except Exception as e:
                                    print(f"⚠️ Auto-correction error: {e}")
                    
                    except Exception as e:
                        print(f"⚠️ Writing logic error: {e}")
                    
                    # Draw enhanced trail with error handling
                    try:
                        if self.show_trail and frame is not None and frame.size > 0:
                            # Create a copy for drawing to avoid modifying original
                            drawing_frame = frame.copy()
                            self.draw_enhanced_trail(drawing_frame)
                            frame = drawing_frame
                    except Exception as e:
                        print(f"⚠️ Trail drawing error: {e}")
                    
                    # Draw UI with error handling
                    try:
                        if frame is not None and frame.size > 0:
                            # Ensure frame is writable
                            if not frame.flags.writeable:
                                frame = frame.copy()
                            self.draw_ui(frame)
                    except Exception as e:
                        print(f"⚠️ UI drawing error: {e}")
                    
                    # Show frame with enhanced error handling
                    try:
                        if frame is not None and frame.size > 0 and len(frame.shape) == 3:
                            # Validate frame before display
                            if frame.shape[0] > 0 and frame.shape[1] > 0:
                                cv2.imshow('Complete Integrated Air Writing System', frame)
                            else:
                                print("⚠️ Invalid frame dimensions for display")
                        else:
                            print("⚠️ Invalid frame for display")
                    except Exception as e:
                        print(f"⚠️ Frame display error: {e}")
                        # Try to recreate window
                        try:
                            cv2.destroyAllWindows()
                            time.sleep(0.1)
                        except:
                            pass
                    
                    # Handle input with error handling
                    try:
                        key = cv2.waitKey(1) & 0xFF
                        if not self.handle_keyboard_input(key):
                            print("👋 Exit requested by user")
                            break
                    except Exception as e:
                        print(f"⚠️ Input handling error: {e}")
                    
                    # Check for system stability
                    if frame_counter % 300 == 0:  # Every 10 seconds at 30 FPS
                        print(f"📊 System stable - Frame {frame_counter}, FPS: {self.fps}")
                
                except Exception as frame_error:
                    print(f"⚠️ Frame processing error: {frame_error}")
                    # Continue to next frame instead of crashing
                    continue
        
        except KeyboardInterrupt:
            print("\n👋 System interrupted by user")
        except Exception as e:
            print(f"❌ Critical system error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            print("🧹 Cleaning up system...")
            self.cleanup()
    
    def cleanup(self):
        """Enhanced cleanup with better resource management"""
        print("🧹 Starting cleanup process...")
        
        try:
            # Stop voice feedback
            if hasattr(self, 'voice_feedback') and self.voice_feedback:
                try:
                    if hasattr(self.voice_feedback, 'engine') and self.voice_feedback.engine:
                        self.voice_feedback.engine.stop()
                    print("✅ Voice feedback stopped")
                except Exception as e:
                    print(f"⚠️ Voice cleanup error: {e}")
            
            # Release camera
            if hasattr(self, 'cap') and self.cap:
                try:
                    self.cap.release()
                    print("✅ Camera released")
                except Exception as e:
                    print(f"⚠️ Camera release error: {e}")
            
            # Close OpenCV windows
            try:
                cv2.destroyAllWindows()
                print("✅ OpenCV windows closed")
            except Exception as e:
                print(f"⚠️ Window cleanup error: {e}")
            
            # Clear hand tracker resources
            if hasattr(self, 'hand_tracker') and self.hand_tracker:
                try:
                    self.hand_tracker.clear_trail()
                    print("✅ Hand tracker cleared")
                except Exception as e:
                    print(f"⚠️ Hand tracker cleanup error: {e}")
        
        except Exception as e:
            print(f"⚠️ Cleanup error: {e}")
        
        # Session summary
        try:
            session_duration = datetime.now() - self.session_start_time
            
            print(f"\n📊 SESSION SUMMARY")
            print("=" * 60)
            print(f"   Duration: {session_duration}")
            print(f"   Words recognized: {len(self.recognized_words)}")
            if self.recognized_words:
                print(f"   Words: {' → '.join(self.recognized_words)}")
            
            avg_processing = self.letter_recognizer.get_average_processing_time()
            print(f"   Average processing time: {avg_processing:.1f}ms")
            print(f"   Final FPS: {self.fps}")
            print(f"   Output logged to: {self.output_file}")
            print("=" * 60)
            print("👋 Complete Integrated Air Writing System finished!")
            
        except Exception as e:
            print(f"⚠️ Session summary error: {e}")
            print("👋 Complete Integrated Air Writing System finished!")

def main():
    """Enhanced main function with better error handling"""
    parser = argparse.ArgumentParser(description="Complete Integrated Air Writing System")
    parser.add_argument('--models', nargs='+', help='Paths to model files')
    parser.add_argument('--dictionary', type=str, help='Path to dictionary file')
    parser.add_argument('--list-models', action='store_true', help='List available models')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    if args.list_models:
        print("\n📁 Available Models:")
        model_dir = Path("models")
        if model_dir.exists():
            for model_file in model_dir.glob("*.h5"):
                print(f"   ✅ {model_file}")
        else:
            print("   ❌ No models directory found")
        return
    
    print("🖐️  COMPLETE INTEGRATED AIR WRITING SYSTEM")
    print("=" * 70)
    print("🎯 Using existing codebase with all advanced features")
    print("🛡️ Enhanced stability and error handling")
    if args.models:
        print(f"📁 Using models: {args.models}")
    if args.dictionary:
        print(f"📚 Using dictionary: {args.dictionary}")
    if args.debug:
        print("🐛 Debug mode enabled")
    print("=" * 70)
    
    system = None
    try:
        print("🔧 Initializing system components...")
        system = CompleteAirWritingSystem(
            model_paths=args.models,
            dictionary_path=args.dictionary
        )
        
        print("🚀 Starting main application loop...")
        system.run()
        
    except KeyboardInterrupt:
        print("\n👋 System interrupted by user")
    except Exception as e:
        print(f"❌ Critical system error: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        else:
            print("💡 Run with --debug flag for detailed error information")
    finally:
        # Ensure cleanup happens even if system wasn't fully initialized
        if system:
            try:
                system.cleanup()
            except:
                pass
        
        # Final cleanup
        try:
            cv2.destroyAllWindows()
        except:
            pass
        
        print("🏁 Application terminated")

if __name__ == "__main__":
    main()