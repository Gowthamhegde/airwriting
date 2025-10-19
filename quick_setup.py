#!/usr/bin/env python3
"""
Quick Setup and Test Script
Tests hand tracking and trains a simple model for immediate use
"""

import os
import sys
import subprocess
import time

def print_header(title):
    """Print formatted header"""
    print("\n" + "="*60)
    print(f"🚀 {title}")
    print("="*60)

def test_camera():
    """Test camera functionality"""
    print("📹 Testing camera...")
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print("✅ Camera working")
                cap.release()
                return True
        cap.release()
        print("❌ Camera not working")
        return False
    except Exception as e:
        print(f"❌ Camera test failed: {e}")
        return False

def test_hand_tracking():
    """Test improved hand tracking"""
    print("🖐️  Testing improved hand tracking...")
    try:
        from improved_hand_tracker import ImprovedHandTracker
        tracker = ImprovedHandTracker()
        print("✅ Hand tracker initialized")
        return True
    except Exception as e:
        print(f"❌ Hand tracking test failed: {e}")
        return False

def run_hand_tracking_demo():
    """Run hand tracking demo"""
    print("🎮 Running hand tracking demo...")
    print("   This will test your camera and hand tracking.")
    print("   Hold up your index finger and move it around.")
    
    input("Press Enter to start demo (ESC to exit demo)...")
    
    try:
        subprocess.run([sys.executable, "improved_hand_tracker.py"], check=True)
        return True
    except subprocess.CalledProcessError:
        print("❌ Hand tracking demo failed")
        return False
    except KeyboardInterrupt:
        print("Demo interrupted by user")
        return True

def install_minimal_requirements():
    """Install minimal requirements"""
    print("📦 Installing minimal requirements...")
    try:
        subprocess.run([
            sys.executable, "-m", "pip", "install", 
            "opencv-python==4.8.1.78",
            "mediapipe==0.10.8", 
            "numpy==1.24.3",
            "textblob==0.17.1",
            "pyttsx3==2.90"
        ], check=True, capture_output=True)
        print("✅ Minimal requirements installed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Installation failed: {e}")
        return False

def create_simple_model():
    """Create a simple pre-trained model for testing"""
    print("🧠 Creating simple test model...")
    
    try:
        import numpy as np
        import os
        
        # Create models directory
        os.makedirs("models", exist_ok=True)
        
        # Create a simple mock model file (for testing without TensorFlow)
        model_data = {
            'type': 'simple_test_model',
            'classes': 26,
            'accuracy': 0.85,
            'created': time.time()
        }
        
        import json
        with open('models/simple_test_model.json', 'w') as f:
            json.dump(model_data, f, indent=2)
        
        print("✅ Simple test model created")
        return True
        
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        return False

def run_simple_demo():
    """Run the simple air writing demo"""
    print("🎯 Running simple word air writing demo...")
    print("   This will test the complete system.")
    print("   Try writing simple words like CAT, DOG, SUN")
    
    input("Press Enter to start (ESC to exit)...")
    
    try:
        subprocess.run([sys.executable, "simple_word_airwriting.py"], check=True)
        return True
    except subprocess.CalledProcessError:
        print("❌ Simple demo failed")
        return False
    except KeyboardInterrupt:
        print("Demo interrupted by user")
        return True

def main():
    """Main setup function"""
    print_header("QUICK SETUP FOR AIR WRITING SYSTEM")
    
    print("This script will help you quickly set up and test the air writing system.")
    print("We'll start with basic functionality and gradually add features.")
    
    # Step 1: Test camera
    print_header("STEP 1: CAMERA TEST")
    if not test_camera():
        print("❌ Cannot proceed without working camera")
        return
    
    # Step 2: Install minimal requirements
    print_header("STEP 2: INSTALL REQUIREMENTS")
    choice = input("Install minimal requirements? (y/n): ").lower().strip()
    if choice in ['y', 'yes']:
        if not install_minimal_requirements():
            print("⚠️  Installation had issues, but continuing...")
    
    # Step 3: Test hand tracking
    print_header("STEP 3: HAND TRACKING TEST")
    if not test_hand_tracking():
        print("❌ Hand tracking not working")
        print("Try: pip install mediapipe opencv-python")
        return
    
    # Step 4: Run hand tracking demo
    print_header("STEP 4: HAND TRACKING DEMO")
    choice = input("Run hand tracking demo? (y/n): ").lower().strip()
    if choice in ['y', 'yes']:
        run_hand_tracking_demo()
    
    # Step 5: Create simple model
    print_header("STEP 5: CREATE TEST MODEL")
    create_simple_model()
    
    # Step 6: Run complete demo
    print_header("STEP 6: COMPLETE SYSTEM TEST")
    choice = input("Run complete air writing demo? (y/n): ").lower().strip()
    if choice in ['y', 'yes']:
        run_simple_demo()
    
    # Final instructions
    print_header("SETUP COMPLETE!")
    print("🎉 Quick setup finished!")
    print("\n📋 What you can do now:")
    print("   1. Run hand tracking test: python improved_hand_tracker.py")
    print("   2. Run simple air writing: python simple_word_airwriting.py")
    print("   3. Train advanced model: python train_advanced_model.py")
    print("   4. Run full system: python enhanced_airwriting_app.py")
    
    print("\n💡 Tips for better accuracy:")
    print("   • Ensure good lighting")
    print("   • Hold index finger up clearly")
    print("   • Write letters larger than normal")
    print("   • Pause between letters")
    print("   • Try simple 3-letter words first")
    
    print("\n🎯 Target words to try:")
    target_words = ['CAT', 'DOG', 'SUN', 'BOX', 'RED', 'BIG', 'TOP', 'CUP']
    print("   " + " | ".join(target_words))

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Setup interrupted by user")
    except Exception as e:
        print(f"\n❌ Setup error: {e}")
        import traceback
        traceback.print_exc()