"""
Enhanced Word Recognition Module
Provides advanced word correction and recognition capabilities
"""

import re
from textblob import TextBlob
from collections import Counter
import difflib

class WordRecognizer:
    def __init__(self):
        # Common English words for better correction
        self.common_words = {
            'HELLO', 'WORLD', 'PYTHON', 'CODE', 'WRITE', 'AIR', 'HAND', 'FINGER',
            'COMPUTER', 'CAMERA', 'VISION', 'MACHINE', 'LEARNING', 'DEEP', 'NEURAL',
            'NETWORK', 'MODEL', 'TRAIN', 'TEST', 'DATA', 'IMAGE', 'PROCESS',
            'ALGORITHM', 'SYSTEM', 'REAL', 'TIME', 'RECOGNITION', 'GESTURE',
            'TRACKING', 'DETECTION', 'CLASSIFICATION', 'PREDICTION', 'ACCURACY',
            'PERFORMANCE', 'OPTIMIZATION', 'FEATURE', 'EXTRACTION', 'PREPROCESSING',
            'CAT', 'DOG', 'BAT', 'HAT', 'MAT', 'RAT', 'SAT', 'FAT', 'PAT',
            'BAG', 'TAG', 'WAG', 'BIG', 'DIG', 'FIG', 'GIG', 'JIG', 'PIG', 'RIG', 'WIG',
            'BUG', 'DUG', 'HUG', 'JUG', 'MUG', 'PUG', 'RUG', 'TUG',
            'CUB', 'HUB', 'NUB', 'PUB', 'RUB', 'SUB', 'TUB'
        }
        
        # Letter confusion matrix (common OCR/handwriting errors)
        self.confusion_pairs = {
            'O': ['0', 'Q', 'D'],
            'I': ['1', 'L', 'J'],
            'S': ['5', 'Z'],
            'B': ['8', 'D'],
            'G': ['6', 'C'],
            'Z': ['2', 'S'],
            'A': ['4', 'R'],
            'E': ['3', 'F'],
            'T': ['7', 'F', 'I'],
            'U': ['V', 'Y'],
            'V': ['U', 'Y'],
            'W': ['VV', 'M'],
            'M': ['N', 'W'],
            'N': ['M', 'H'],
            'P': ['R', 'B'],
            'R': ['P', 'A'],
            'C': ['G', 'O'],
            'D': ['O', 'B'],
            'F': ['E', 'T'],
            'H': ['N', 'M'],
            'J': ['I', 'L'],
            'K': ['X', 'R'],
            'L': ['I', 'J'],
            'Q': ['O', 'G'],
            'X': ['K', 'Y'],
            'Y': ['V', 'X']
        }
    
    def preprocess_word(self, word):
        """Clean and preprocess the input word"""
        if not word:
            return ""
        
        # Remove non-alphabetic characters and convert to uppercase
        cleaned = re.sub(r'[^A-Za-z]', '', word).upper()
        return cleaned
    
    def generate_candidates(self, word):
        """Generate candidate corrections for a word"""
        candidates = set()
        
        # Original word
        candidates.add(word)
        
        # Single character substitutions using confusion matrix
        for i, char in enumerate(word):
            if char in self.confusion_pairs:
                for replacement in self.confusion_pairs[char]:
                    if len(replacement) == 1:
                        candidate = word[:i] + replacement + word[i+1:]
                        candidates.add(candidate)
        
        # Single character deletions
        for i in range(len(word)):
            candidate = word[:i] + word[i+1:]
            if candidate:
                candidates.add(candidate)
        
        # Single character insertions
        alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        for i in range(len(word) + 1):
            for char in alphabet:
                candidate = word[:i] + char + word[i:]
                candidates.add(candidate)
        
        # Adjacent character swaps
        for i in range(len(word) - 1):
            candidate = word[:i] + word[i+1] + word[i] + word[i+2:]
            candidates.add(candidate)
        
        return candidates
    
    def calculate_similarity(self, word1, word2):
        """Calculate similarity between two words"""
        return difflib.SequenceMatcher(None, word1, word2).ratio()
    
    def correct_with_dictionary(self, word):
        """Correct word using custom dictionary and common words"""
        word = self.preprocess_word(word)
        
        if not word:
            return ""
        
        # Check if word is already in common words
        if word in self.common_words:
            return word
        
        # Generate candidates and find best match
        candidates = self.generate_candidates(word)
        
        # Filter candidates that exist in common words
        valid_candidates = [c for c in candidates if c in self.common_words]
        
        if valid_candidates:
            # Return the candidate with highest similarity to original
            best_candidate = max(valid_candidates, 
                               key=lambda x: self.calculate_similarity(word, x))
            return best_candidate
        
        # If no valid candidates, try TextBlob correction
        try:
            blob = TextBlob(word.lower())
            corrected = str(blob.correct()).upper()
            return corrected
        except:
            return word
    
    def correct_word_advanced(self, word):
        """Advanced word correction with multiple strategies"""
        if not word:
            return ""
        
        original_word = word
        word = self.preprocess_word(word)
        
        # Strategy 1: Direct dictionary lookup
        if word in self.common_words:
            return word
        
        # Strategy 2: Custom dictionary correction
        dict_corrected = self.correct_with_dictionary(word)
        if dict_corrected != word and dict_corrected in self.common_words:
            return dict_corrected
        
        # Strategy 3: TextBlob correction
        try:
            blob = TextBlob(word.lower())
            textblob_corrected = str(blob.correct()).upper()
            
            # Prefer TextBlob if it's in our common words
            if textblob_corrected.upper() in self.common_words:
                return textblob_corrected.upper()
            
            # Otherwise, compare similarities
            dict_similarity = self.calculate_similarity(word, dict_corrected)
            textblob_similarity = self.calculate_similarity(word, textblob_corrected.upper())
            
            if textblob_similarity > dict_similarity:
                return textblob_corrected.upper()
            else:
                return dict_corrected
                
        except:
            return dict_corrected
    
    def suggest_completions(self, partial_word):
        """Suggest word completions for partial input"""
        if not partial_word:
            return []
        
        partial = self.preprocess_word(partial_word)
        suggestions = []
        
        for word in self.common_words:
            if word.startswith(partial):
                suggestions.append(word)
        
        # Sort by length (shorter completions first)
        suggestions.sort(key=len)
        return suggestions[:5]  # Return top 5 suggestions
    
    def analyze_word_confidence(self, word):
        """Analyze confidence in word recognition"""
        word = self.preprocess_word(word)
        
        if not word:
            return 0.0
        
        confidence_factors = []
        
        # Factor 1: Word length (favor short words if in dictionary)
        if len(word) <= 3:
            length_factor = 0.8  # Boost for short words
        else:
            length_factor = min(len(word) / 8.0, 1.0)  # Normalize to max 8 chars
        confidence_factors.append(length_factor)
        
        # Factor 2: Dictionary presence
        dict_factor = 1.0 if word in self.common_words else 0.3
        confidence_factors.append(dict_factor)
        
        # Factor 3: Character consistency (no repeated unusual patterns)
        pattern_factor = 1.0
        for i in range(len(word) - 1):
            if word[i] == word[i + 1] and word[i] in 'IIILLL':  # Common OCR errors
                pattern_factor *= 0.8
        confidence_factors.append(pattern_factor)
        
        # Calculate weighted average
        weights = [0.2, 0.6, 0.2]  # Dictionary presence is most important
        confidence = sum(f * w for f, w in zip(confidence_factors, weights))
        
        return min(confidence, 1.0)

# Global instance for easy import
word_recognizer = WordRecognizer()

def correct_word_enhanced(word):
    """Enhanced word correction function"""
    return word_recognizer.correct_word_advanced(word)

def get_word_suggestions(partial_word):
    """Get word completion suggestions"""
    return word_recognizer.suggest_completions(partial_word)

def get_word_confidence(word):
    """Get confidence score for a word"""
    return word_recognizer.analyze_word_confidence(word)