#!/usr/bin/env python3
"""
Setup script for Real-Time Air Writing System
Installs dependencies and initializes the system
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    print("📦 Installing required packages...")
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements_realtime.txt"
        ])
        print("✅ All packages installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing packages: {e}")
        return False

def download_textblob_corpora():
    """Download TextBlob corpora for auto-correction"""
    print("📚 Downloading TextBlob corpora...")
    
    try:
        import textblob
        textblob.download_corpora.download_all()
        print("✅ TextBlob corpora downloaded!")
        return True
    except Exception as e:
        print(f"⚠️ Warning: Could not download TextBlob corpora: {e}")
        return False

def test_camera():
    """Test camera access"""
    print("🎥 Testing camera access...")
    
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print("✅ Camera test successful!")
                cap.release()
                return True
            else:
                print("❌ Could not read from camera")
                cap.release()
                return False
        else:
            print("❌ Could not open camera")
            return False
    except Exception as e:
        print(f"❌ Camera test failed: {e}")
        return False

def test_audio():
    """Test text-to-speech"""
    print("🔊 Testing text-to-speech...")
    
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.say("Audio test successful")
        engine.runAndWait()
        print("✅ Audio test successful!")
        return True
    except Exception as e:
        print(f"⚠️ Warning: Audio test failed: {e}")
        print("   The system will work but without voice feedback")
        return False

def main():
    """Main setup function"""
    print("🖐️ REAL-TIME AIR WRITING SYSTEM SETUP")
    print("=" * 50)
    
    success_count = 0
    total_tests = 4
    
    # Install requirements
    if install_requirements():
        success_count += 1
    
    # Download TextBlob corpora
    if download_textblob_corpora():
        success_count += 1
    
    # Test camera
    if test_camera():
        success_count += 1
    
    # Test audio
    if test_audio():
        success_count += 1
    
    print("\n" + "=" * 50)
    print(f"📊 Setup Results: {success_count}/{total_tests} tests passed")
    
    if success_count >= 3:
        print("🎉 Setup completed successfully!")
        print("\n🚀 To run the system:")
        print("   python realtime_airwriting_system.py")
        print("\n📋 Instructions:")
        print("   • Open hand = Start tracking")
        print("   • Close hand (fist) = Stop & recognize")
        print("   • Press 'C' = Clear canvas")
        print("   • Press 'ESC' = Exit")
    else:
        print("⚠️ Setup completed with warnings")
        print("   Some features may not work properly")
        print("   Please check the error messages above")

if __name__ == "__main__":
    main()