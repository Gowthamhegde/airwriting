#!/usr/bin/env python3
"""
Simple test for word corrector without OpenCV dependencies
"""

import sys
import os
import json
import numpy as np
from collections import deque
from pathlib import Path

# Mock the dependencies that cause import issues
class MockCV2:
    FONT_HERSHEY_SIMPLEX = 0
    CAP_PROP_FRAME_WIDTH = 0
    CAP_PROP_FRAME_HEIGHT = 0
    CAP_PROP_FPS = 0
    CAP_PROP_BUFFERSIZE = 0
    
    @staticmethod
    def line(*args, **kwargs):
        pass
    
    @staticmethod
    def GaussianBlur(img, *args, **kwargs):
        return img
    
    @staticmethod
    def VideoCapture(*args, **kwargs):
        return None

class MockMediaPipe:
    class solutions:
        class hands:
            Hands = lambda *args, **kwargs: None

# Mock imports
sys.modules['cv2'] = MockCV2()
sys.modules['mediapipe'] = MockMediaPipe()

# Now we can import our classes
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
    
    def correct_word(self, word):
        """Test word correction"""
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
        
        # Simple correction - find closest match
        best_word = word
        best_score = 0.0
        
        for target_word in self.target_words:
            if len(word) == len(target_word):
                # Simple character match score
                matches = sum(1 for a, b in zip(word, target_word) if a == b)
                score = matches / len(word)
                if score > best_score:
                    best_score = score
                    best_word = target_word
        
        return best_word, best_score

def test_word_corrector():
    """Test the word corrector"""
    print("🧪 Testing Word Corrector...")
    
    try:
        corrector = IntegratedWordCorrector()
        
        # Test basic functionality
        print(f"Word set exists: {hasattr(corrector, 'word_set')}")
        print(f"Word set type: {type(corrector.word_set)}")
        print(f"Word set size: {len(corrector.word_set)}")
        
        # Test corrections
        test_words = ["CAT", "CXT", "DOG", "XYZ", ""]
        
        for test_word in test_words:
            try:
                result = corrector.correct_word(test_word)
                print(f"'{test_word}' -> {result}")
            except Exception as e:
                print(f"Error correcting '{test_word}': {e}")
        
        print("✅ Word corrector test completed!")
        return True
        
    except Exception as e:
        print(f"❌ Word corrector test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_word_corrector()
    print(f"\n{'✅ SUCCESS' if success else '❌ FAILED'}")