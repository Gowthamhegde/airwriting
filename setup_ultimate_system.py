#!/usr/bin/env python3
"""
Ultimate Air Writing System Setup
Comprehensive setup and installation script for the complete system
"""

import os
import sys
import subprocess
import platform
import time
from pathlib import Path

class UltimateSystemSetup:
    """Setup manager for the ultimate air writing system"""
    
    def __init__(self):
        self.system_os = platform.system()
        self.python_version = sys.version_info
        self.project_root = Path(__file__).parent
        
        # Required packages
        self.core_packages = [
            'opencv-python>=4.8.0',
            'mediapipe>=0.10.0',
            'numpy>=1.21.0',
        ]
        
        self.ml_packages = [
            'tensorflow>=2.10.0',
            'scikit-learn>=1.1.0',
        ]
        
        self.text_packages = [
            'textblob>=0.17.1',
        ]
        
        self.audio_packages = [
            'pyttsx3>=2.90',
        ]
        
        self.optional_packages = [
            'scipy>=1.9.0',
            'matplotlib>=3.5.0',
            'Pillow>=9.0.0',
        ]
        
        print("🚀 Ultimate Air Writing System Setup")
        print(f"OS: {self.system_os}")
        print(f"Python: {self.python_version.major}.{self.python_version.minor}.{self.python_version.micro}")
    
    def check_python_version(self):
        """Check if Python version is compatible"""
        print("\n🐍 Checking Python version...")
        
        if self.python_version.major < 3 or (self.python_version.major == 3 and self.python_version.minor < 8):
            print("❌ Python 3.8 or higher is required")
            print(f"   Current version: {self.python_version.major}.{self.python_version.minor}.{self.python_version.micro}")
            return False
        
        print(f"✅ Python version compatible: {self.python_version.major}.{self.python_version.minor}.{self.python_version.micro}")
        return True
    
    def install_packages(self, packages, category_name, required=True):
        """Install a list of packages"""
        print(f"\n📦 Installing {category_name} packages...")
        
        failed_packages = []
        
        for package in packages:
            try:
                print(f"   Installing {package}...")
                result = subprocess.run([
                    sys.executable, '-m', 'pip', 'install', package
                ], capture_output=True, text=True, timeout=300)
                
                if result.returncode == 0:
                    print(f"   ✅ {package} installed successfully")
                else:
                    print(f"   ❌ Failed to install {package}")
                    print(f"      Error: {result.stderr}")
                    failed_packages.append(package)
                    
            except subprocess.TimeoutExpired:
                print(f"   ⏰ Timeout installing {package}")
                failed_packages.append(package)
            except Exception as e:
                print(f"   ❌ Error installing {package}: {e}")
                failed_packages.append(package)
        
        if failed_packages:
            print(f"\n⚠️  Failed to install {len(failed_packages)} {category_name} packages:")
            for package in failed_packages:
                print(f"   • {package}")
            
            if required:
                print(f"❌ {category_name} packages are required for the system to work")
                return False
            else:
                print(f"⚠️  {category_name} packages are optional - system will work with reduced functionality")
        else:
            print(f"✅ All {category_name} packages installed successfully")
        
        return len(failed_packages) == 0
    
    def test_imports(self):
        """Test if all packages can be imported"""
        print("\n🧪 Testing package imports...")
        
        test_results = {}
        
        # Test core packages
        core_imports = [
            ('cv2', 'OpenCV'),
            ('mediapipe', 'MediaPipe'),
            ('numpy', 'NumPy'),
        ]
        
        ml_imports = [
            ('tensorflow', 'TensorFlow'),
            ('sklearn', 'Scikit-learn'),
        ]
        
        text_imports = [
            ('textblob', 'TextBlob'),
        ]
        
        audio_imports = [
            ('pyttsx3', 'Text-to-Speech'),
        ]
        
        all_imports = [
            (core_imports, 'Core', True),
            (ml_imports, 'Machine Learning', False),
            (text_imports, 'Text Processing', False),
            (audio_imports, 'Audio', False),
        ]
        
        for imports, category, required in all_imports:
            print(f"\n   {category} packages:")
            category_success = True
            
            for module, name in imports:
                try:
                    __import__(module)
                    print(f"   ✅ {name}")
                    test_results[module] = True
                except ImportError as e:
                    print(f"   ❌ {name} - {e}")
                    test_results[module] = False
                    category_success = False
            
            if not category_success and required:
                print(f"   ❌ {category} packages are required!")
                return False
        
        return True
    
    def create_directories(self):
        """Create necessary directories"""
        print("\n📁 Creating project directories...")
        
        directories = [
            'models',
            'modules',
            'utils',
            'logs',
            'output',
        ]
        
        for directory in directories:
            dir_path = self.project_root / directory
            try:
                dir_path.mkdir(exist_ok=True)
                print(f"   ✅ {directory}/")
            except Exception as e:
                print(f"   ❌ Failed to create {directory}/: {e}")
                return False
        
        return True
    
    def create_sample_files(self):
        """Create sample configuration files"""
        print("\n📄 Creating sample configuration files...")
        
        # Create sample dictionary
        sample_dictionary = {
            "target_words": [
                "THE", "AND", "FOR", "ARE", "BUT", "NOT", "YOU", "ALL", "CAN", "HER",
                "CAT", "DOG", "BAT", "RAT", "HAT", "MAT", "SAT", "FAT", "PAT", "VAT",
                "RUN", "SIT", "EAT", "SEE", "HIT", "CUT", "DIG", "FLY", "WIN", "TRY",
                "HELLO", "WORLD", "WRITE", "LETTER", "WORD", "SYSTEM", "COMPUTER"
            ],
            "word_frequencies": {}
        }
        
        # Add frequencies
        for i, word in enumerate(sample_dictionary["target_words"]):
            freq = 1.0 - (i * 0.02)
            sample_dictionary["word_frequencies"][word] = max(0.1, freq)
        
        try:
            import json
            dict_path = self.project_root / "models" / "sample_dictionary.json"
            with open(dict_path, 'w') as f:
                json.dump(sample_dictionary, f, indent=2)
            print("   ✅ Sample dictionary created")
        except Exception as e:
            print(f"   ❌ Failed to create sample dictionary: {e}")
        
        # Create README
        readme_content = """# Ultimate Air Writing Recognition System

## Quick Start

1. Run the setup: `python setup_ultimate_system.py`
2. Start the system: `python ultimate_airwriting_system.py`

## Hand Gestures

- ✍️ INDEX FINGER UP (others curled) = Writing mode
- ✋ OPEN HAND (all fingers extended) = Pause tracking
- ✊ CLOSED HAND (fist) = Pause tracking
- ✌️ PEACE SIGN = Clear current word
- 👌 OK SIGN = Complete current word

## Keyboard Controls

- SPACE: Force complete current letter
- ENTER: Force complete current word
- C: Clear current word
- R: Reset everything
- T: Toggle trail display
- S: Toggle word suggestions
- D: Toggle debug information
- V: Test voice feedback
- ESC: Exit system

## Files

- `ultimate_airwriting_system.py` - Main system
- `enhanced_realtime_airwriting.py` - Alternative implementation
- `modules/advanced_hand_detection.py` - Hand gesture detection
- `modules/ensemble_letter_recognition.py` - Letter recognition
- `models/` - Trained models and dictionaries
- `logs/` - System logs and output

## Troubleshooting

1. Camera not working: Check camera permissions and connections
2. No voice feedback: Install pyttsx3 or check audio settings
3. Poor recognition: Train custom models or adjust confidence thresholds
4. Performance issues: Reduce camera resolution or disable debug mode
"""
        
        try:
            readme_path = self.project_root / "README_ULTIMATE.md"
            with open(readme_path, 'w') as f:
                f.write(readme_content)
            print("   ✅ README created")
        except Exception as e:
            print(f"   ❌ Failed to create README: {e}")
    
    def test_camera(self):
        """Test camera functionality"""
        print("\n📹 Testing camera...")
        
        try:
            import cv2
            cap = cv2.VideoCapture(0)
            
            if not cap.isOpened():
                print("   ❌ Camera not accessible")
                return False
            
            ret, frame = cap.read()
            if not ret or frame is None:
                print("   ❌ Cannot read from camera")
                cap.release()
                return False
            
            print(f"   ✅ Camera working - Resolution: {frame.shape[1]}x{frame.shape[0]}")
            cap.release()
            return True
            
        except Exception as e:
            print(f"   ❌ Camera test failed: {e}")
            return False
    
    def test_system_components(self):
        """Test individual system components"""
        print("\n🧪 Testing system components...")
        
        # Test hand detection
        try:
            from modules.advanced_hand_detection import AdvancedHandDetector
            detector = AdvancedHandDetector()
            print("   ✅ Hand detection module")
        except Exception as e:
            print(f"   ❌ Hand detection module: {e}")
        
        # Test letter recognition
        try:
            from modules.ensemble_letter_recognition import EnsembleLetterRecognizer
            recognizer = EnsembleLetterRecognizer()
            print("   ✅ Letter recognition module")
        except Exception as e:
            print(f"   ❌ Letter recognition module: {e}")
        
        # Test main system
        try:
            from ultimate_airwriting_system import UltimateWordCorrector, UltimateVoiceFeedback
            corrector = UltimateWordCorrector()
            voice = UltimateVoiceFeedback()
            print("   ✅ Main system components")
        except Exception as e:
            print(f"   ❌ Main system components: {e}")
    
    def run_full_setup(self):
        """Run complete setup process"""
        print("🚀 Starting Ultimate Air Writing System Setup...")
        print("=" * 60)
        
        # Check Python version
        if not self.check_python_version():
            return False
        
        # Install packages
        print("\n📦 Installing required packages...")
        
        # Core packages (required)
        if not self.install_packages(self.core_packages, "Core", required=True):
            print("❌ Failed to install core packages - setup aborted")
            return False
        
        # ML packages (optional but recommended)
        self.install_packages(self.ml_packages, "Machine Learning", required=False)
        
        # Text processing packages
        self.install_packages(self.text_packages, "Text Processing", required=False)
        
        # Audio packages
        self.install_packages(self.audio_packages, "Audio", required=False)
        
        # Optional packages
        self.install_packages(self.optional_packages, "Optional", required=False)
        
        # Test imports
        if not self.test_imports():
            print("❌ Package import tests failed")
            return False
        
        # Create directories
        if not self.create_directories():
            print("❌ Failed to create directories")
            return False
        
        # Create sample files
        self.create_sample_files()
        
        # Test camera
        camera_ok = self.test_camera()
        
        # Test system components
        self.test_system_components()
        
        # Final summary
        print("\n" + "=" * 60)
        print("🎉 SETUP COMPLETE!")
        print("=" * 60)
        
        if camera_ok:
            print("✅ All systems ready!")
            print("\n🚀 To start the Ultimate Air Writing System:")
            print("   python ultimate_airwriting_system.py")
            print("\n🎯 Alternative systems:")
            print("   python enhanced_realtime_airwriting.py")
            print("   python air_writing_system.py")
        else:
            print("⚠️  Setup complete but camera issues detected")
            print("   Please check camera connections and permissions")
        
        print("\n📚 For help and documentation:")
        print("   python ultimate_airwriting_system.py --help")
        print("   Check README_ULTIMATE.md")
        
        return True
    
    def quick_test(self):
        """Run quick system test"""
        print("🧪 Running quick system test...")
        
        try:
            # Test basic imports
            import cv2
            import mediapipe as mp
            import numpy as np
            print("✅ Core packages working")
            
            # Test camera
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    print("✅ Camera working")
                else:
                    print("⚠️  Camera accessible but cannot read frames")
                cap.release()
            else:
                print("❌ Camera not accessible")
            
            # Test MediaPipe
            hands = mp.solutions.hands.Hands()
            print("✅ MediaPipe hands initialized")
            
            print("🎉 Quick test passed!")
            return True
            
        except Exception as e:
            print(f"❌ Quick test failed: {e}")
            return False

def main():
    """Main setup function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Ultimate Air Writing System Setup")
    parser.add_argument('--quick-test', action='store_true', help='Run quick system test only')
    parser.add_argument('--install-only', action='store_true', help='Install packages only')
    
    args = parser.parse_args()
    
    setup = UltimateSystemSetup()
    
    if args.quick_test:
        setup.quick_test()
    elif args.install_only:
        setup.check_python_version()
        setup.install_packages(setup.core_packages, "Core", required=True)
        setup.install_packages(setup.ml_packages, "Machine Learning", required=False)
        setup.install_packages(setup.text_packages, "Text Processing", required=False)
        setup.install_packages(setup.audio_packages, "Audio", required=False)
        setup.test_imports()
    else:
        success = setup.run_full_setup()
        if not success:
            print("\n❌ Setup failed - please check errors above")
            sys.exit(1)

if __name__ == "__main__":
    main()