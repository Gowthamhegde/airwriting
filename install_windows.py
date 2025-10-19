#!/usr/bin/env python3
"""
Windows-specific installation script for Air Writing Recognition System
Handles common Windows installation issues and dependencies
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def print_system_info():
    """Print system information"""
    print("🖥️  System Information:")
    print(f"   OS: {platform.system()} {platform.release()}")
    print(f"   Architecture: {platform.machine()}")
    print(f"   Python: {sys.version}")
    print(f"   Python executable: {sys.executable}")

def check_pip():
    """Ensure pip is up to date"""
    print("\n📦 Updating pip...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], 
                      check=True, capture_output=True)
        print("✅ pip updated successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"⚠️  pip update failed: {e}")
        return False

def install_core_dependencies():
    """Install core dependencies one by one"""
    print("\n📦 Installing core dependencies...")
    
    # Core dependencies in order of importance
    dependencies = [
        "numpy==1.24.3",
        "opencv-python==4.8.1.78", 
        "pillow==10.0.1",
        "scipy==1.11.4",
        "matplotlib==3.7.2",
        "scikit-learn==1.3.2"
    ]
    
    for dep in dependencies:
        print(f"   Installing {dep}...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", dep], 
                          check=True, capture_output=True)
            print(f"   ✅ {dep} installed")
        except subprocess.CalledProcessError as e:
            print(f"   ❌ Failed to install {dep}")
            return False
    
    return True

def install_mediapipe():
    """Install MediaPipe with fallback versions"""
    print("\n🖐️  Installing MediaPipe...")
    
    # Try different MediaPipe versions
    versions = ["0.10.8", "0.10.7", "0.10.5", "0.10.3"]
    
    for version in versions:
        print(f"   Trying MediaPipe {version}...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", f"mediapipe=={version}"], 
                          check=True, capture_output=True)
            print(f"   ✅ MediaPipe {version} installed successfully")
            return True
        except subprocess.CalledProcessError:
            print(f"   ❌ MediaPipe {version} failed")
            continue
    
    print("❌ Could not install MediaPipe")
    print("💡 Try installing manually:")
    print("   pip install mediapipe --no-deps")
    print("   or download from: https://pypi.org/project/mediapipe/#files")
    return False

def install_tensorflow():
    """Install TensorFlow with CPU support"""
    print("\n🧠 Installing TensorFlow...")
    
    # Try TensorFlow CPU version first
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "tensorflow-cpu==2.13.0"], 
                      check=True, capture_output=True)
        print("✅ TensorFlow CPU installed")
        return True
    except subprocess.CalledProcessError:
        print("⚠️  TensorFlow CPU failed, trying regular TensorFlow...")
        
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "tensorflow==2.13.0"], 
                          check=True, capture_output=True)
            print("✅ TensorFlow installed")
            return True
        except subprocess.CalledProcessError:
            print("❌ TensorFlow installation failed")
            return False

def install_remaining_dependencies():
    """Install remaining dependencies"""
    print("\n📦 Installing remaining dependencies...")
    
    dependencies = [
        "textblob==0.17.1",
        "pyttsx3==2.90",
        "tensorflow-datasets==4.9.3",
        "streamlit==1.28.1"
    ]
    
    for dep in dependencies:
        print(f"   Installing {dep}...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", dep], 
                          check=True, capture_output=True)
            print(f"   ✅ {dep} installed")
        except subprocess.CalledProcessError as e:
            print(f"   ⚠️  {dep} failed, continuing...")
    
    return True

def download_textblob_corpora():
    """Download TextBlob corpora"""
    print("\n📚 Downloading TextBlob corpora...")
    try:
        import textblob
        textblob.download_corpora()
        print("✅ TextBlob corpora downloaded")
        return True
    except Exception as e:
        print(f"⚠️  TextBlob corpora download failed: {e}")
        return False

def test_imports():
    """Test if all critical imports work"""
    print("\n🧪 Testing imports...")
    
    imports = [
        ("cv2", "OpenCV"),
        ("numpy", "NumPy"),
        ("mediapipe", "MediaPipe"),
        ("tensorflow", "TensorFlow"),
        ("textblob", "TextBlob"),
        ("pyttsx3", "Text-to-Speech"),
        ("scipy", "SciPy"),
        ("matplotlib", "Matplotlib"),
        ("sklearn", "Scikit-learn")
    ]
    
    success_count = 0
    for module, name in imports:
        try:
            __import__(module)
            print(f"   ✅ {name}")
            success_count += 1
        except ImportError as e:
            print(f"   ❌ {name}: {e}")
    
    print(f"\n📊 Import test: {success_count}/{len(imports)} successful")
    return success_count >= 6  # Need at least core modules

def create_test_script():
    """Create a simple test script"""
    test_script = """
import cv2
import numpy as np

# Test camera
print("Testing camera...")
cap = cv2.VideoCapture(0)
if cap.isOpened():
    ret, frame = cap.read()
    if ret:
        print("✅ Camera working")
        print(f"   Frame shape: {frame.shape}")
    else:
        print("❌ Camera not providing frames")
    cap.release()
else:
    print("❌ Camera not accessible")

# Test MediaPipe
try:
    import mediapipe as mp
    hands = mp.solutions.hands.Hands()
    print("✅ MediaPipe hands initialized")
except Exception as e:
    print(f"❌ MediaPipe test failed: {e}")

print("\\nTest completed!")
"""
    
    with open("test_system.py", "w") as f:
        f.write(test_script)
    
    print("📝 Created test_system.py - run this to test your setup")

def main():
    """Main installation function"""
    print("🖐️  AIR WRITING RECOGNITION - WINDOWS INSTALLER")
    print("=" * 60)
    
    print_system_info()
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher required")
        return False
    
    steps = [
        ("Update pip", check_pip),
        ("Install core dependencies", install_core_dependencies),
        ("Install MediaPipe", install_mediapipe),
        ("Install TensorFlow", install_tensorflow),
        ("Install remaining dependencies", install_remaining_dependencies),
        ("Download TextBlob corpora", download_textblob_corpora),
        ("Test imports", test_imports)
    ]
    
    for step_name, step_func in steps:
        print(f"\n--- {step_name} ---")
        if not step_func():
            print(f"⚠️  {step_name} had issues, but continuing...")
    
    create_test_script()
    
    print("\n" + "=" * 60)
    print("🎉 Installation completed!")
    print("\nNext steps:")
    print("1. Run: python test_system.py")
    print("2. If tests pass, run: python train_enhanced_model.py")
    print("3. Then run: python demo_airwriting.py")
    print("=" * 60)

if __name__ == "__main__":
    main()