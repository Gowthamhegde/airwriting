#!/usr/bin/env python3
"""
Complete Setup Script for Real-Time Air Writing System
Installs all dependencies and performs system checks
"""

import subprocess
import sys
import os
import time

def print_header():
    """Print setup header"""
    print("🖐️" + "="*58 + "🖐️")
    print("   COMPLETE REAL-TIME AIR WRITING SYSTEM SETUP")
    print("🖐️" + "="*58 + "🖐️")

def install_requirements():
    """Install all required packages"""
    print("\n📦 Installing required packages...")
    
    packages = [
        "opencv-python>=4.8.0",
        "mediapipe>=0.10.0", 
        "numpy>=1.21.0",
        "pyttsx3>=2.90",
        "textblob>=0.17.1",
        "scikit-learn>=1.1.0"
    ]
    
    success_count = 0
    for package in packages:
        try:
            print(f"   Installing {package}...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", package, "--quiet"
            ])
            print(f"   ✅ {package} installed")
            success_count += 1
        except subprocess.CalledProcessError as e:
            print(f"   ❌ Failed to install {package}: {e}")
    
    print(f"\n📊 Package installation: {success_count}/{len(packages)} successful")
    return success_count == len(packages)

def setup_textblob():
    """Setup TextBlob corpora"""
    print("\n📚 Setting up TextBlob for auto-correction...")
    
    try:
        import textblob
        print("   Downloading corpora (this may take a moment)...")
        
        # Download essential corpora
        import nltk
        nltk.download('punkt', quiet=True)
        nltk.download('brown', quiet=True)
        
        print("   ✅ TextBlob setup completed")
        return True
    except Exception as e:
        print(f"   ⚠️ TextBlob setup warning: {e}")
        print("   Auto-correction may not work optimally")
        return False

def test_camera():
    """Test camera functionality"""
    print("\n🎥 Testing camera access...")
    
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("   ❌ Could not open camera")
            return False
        
        # Test frame capture
        ret, frame = cap.read()
        if not ret:
            print("   ❌ Could not capture frame")
            cap.release()
            return False
        
        # Test camera properties
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        cap.release()
        
        print(f"   ✅ Camera working: {int(width)}x{int(height)} @ {int(fps)}fps")
        return True
        
    except Exception as e:
        print(f"   ❌ Camera test failed: {e}")
        return False

def test_mediapipe():
    """Test MediaPipe hands detection"""
    print("\n🖐️ Testing MediaPipe hands detection...")
    
    try:
        import mediapipe as mp
        import cv2
        import numpy as np
        
        # Initialize MediaPipe
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.5
        )
        
        # Create test image with hand-like shape
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.circle(test_image, (320, 240), 100, (255, 255, 255), -1)
        
        # Test processing
        rgb_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_image)
        
        hands.close()
        
        print("   ✅ MediaPipe hands detection ready")
        return True
        
    except Exception as e:
        print(f"   ❌ MediaPipe test failed: {e}")
        return False

def test_audio():
    """Test text-to-speech functionality"""
    print("\n🔊 Testing text-to-speech...")
    
    try:
        import pyttsx3
        
        engine = pyttsx3.init()
        
        # Test basic functionality
        voices = engine.getProperty('voices')
        if voices:
            print(f"   Found {len(voices)} voice(s)")
            
            # Set properties
            engine.setProperty('rate', 150)
            engine.setProperty('volume', 0.8)
            
            # Test speech (brief)
            engine.say("Audio test")
            engine.runAndWait()
            
            print("   ✅ Text-to-speech working")
            return True
        else:
            print("   ⚠️ No voices found, but engine initialized")
            return True
            
    except Exception as e:
        print(f"   ❌ Audio test failed: {e}")
        print("   System will work but without voice feedback")
        return False

def test_machine_learning():
    """Test machine learning components"""
    print("\n🧠 Testing machine learning components...")
    
    try:
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.preprocessing import StandardScaler
        import numpy as np
        
        # Test basic ML functionality
        X = np.random.rand(10, 5)
        y = ['word1'] * 5 + ['word2'] * 5
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        classifier = KNeighborsClassifier(n_neighbors=3)
        classifier.fit(X_scaled, y)
        
        # Test prediction
        test_sample = np.random.rand(1, 5)
        test_scaled = scaler.transform(test_sample)
        prediction = classifier.predict(test_scaled)
        
        print("   ✅ Machine learning components ready")
        return True
        
    except Exception as e:
        print(f"   ❌ ML test failed: {e}")
        return False

def create_demo_script():
    """Create a simple demo script"""
    print("\n📝 Creating demo launcher...")
    
    demo_script = '''#!/usr/bin/env python3
"""
Quick Demo Launcher for Air Writing System
"""

import subprocess
import sys
import os

def main():
    print("🚀 Launching Complete Air Writing System...")
    print("📋 Remember:")
    print("   • Open hand = Start tracking")
    print("   • Close hand = Stop & recognize") 
    print("   • Press C = Clear canvas")
    print("   • Press ESC = Exit")
    print()
    
    try:
        subprocess.run([sys.executable, "complete_realtime_airwriting.py"])
    except KeyboardInterrupt:
        print("\\n👋 Demo ended")
    except FileNotFoundError:
        print("❌ Error: complete_realtime_airwriting.py not found")
        print("Please run setup_complete_system.py first")

if __name__ == "__main__":
    main()
'''
    
    try:
        with open('run_demo.py', 'w') as f:
            f.write(demo_script)
        print("   ✅ Demo launcher created: run_demo.py")
        return True
    except Exception as e:
        print(f"   ❌ Failed to create demo script: {e}")
        return False

def main():
    """Main setup function"""
    print_header()
    
    tests = [
        ("Package Installation", install_requirements),
        ("TextBlob Setup", setup_textblob),
        ("Camera Test", test_camera),
        ("MediaPipe Test", test_mediapipe),
        ("Audio Test", test_audio),
        ("ML Components Test", test_machine_learning),
        ("Demo Script Creation", create_demo_script)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*60}")
        result = test_func()
        results.append((test_name, result))
        time.sleep(0.5)  # Brief pause between tests
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 SETUP SUMMARY")
    print(f"{'='*60}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name:<25} {status}")
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed >= 5:  # Most critical tests passed
        print("\n🎉 SETUP COMPLETED SUCCESSFULLY!")
        print("\n🚀 To run the system:")
        print("   python complete_realtime_airwriting.py")
        print("   OR")
        print("   python run_demo.py")
        
        print("\n📋 Quick Start:")
        print("   1. Run the system")
        print("   2. Show your hand to the camera")
        print("   3. Open hand = start tracking (green trail)")
        print("   4. Close hand = recognize word")
        print("   5. Try words: cat, dog, sun, run, big, etc.")
        
    else:
        print("\n⚠️ SETUP COMPLETED WITH ISSUES")
        print("Some components may not work properly.")
        print("Please check the failed tests above.")
    
    print(f"\n{'='*60}")

if __name__ == "__main__":
    main()