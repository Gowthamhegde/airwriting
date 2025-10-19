#!/usr/bin/env python3
"""
Environment Checker for Air Writing System
Checks Python version, MediaPipe compatibility, and suggests fixes
"""

import sys
import platform
import subprocess
import importlib.util

def print_system_info():
    """Print detailed system information"""
    print("🖥️  SYSTEM INFORMATION")
    print("=" * 50)
    print(f"Operating System: {platform.system()} {platform.release()}")
    print(f"Architecture: {platform.machine()}")
    print(f"Python Version: {sys.version}")
    print(f"Python Executable: {sys.executable}")
    print(f"Python Path: {sys.path[0]}")
    
    # Check if we're in a virtual environment
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("Virtual Environment: Yes")
    else:
        print("Virtual Environment: No")
    
    print("=" * 50)

def check_python_compatibility():
    """Check Python version compatibility"""
    print("\n🐍 PYTHON VERSION CHECK")
    print("-" * 30)
    
    version = sys.version_info
    print(f"Current Python: {version.major}.{version.minor}.{version.micro}")
    
    # MediaPipe compatibility matrix
    compatible_versions = {
        (3, 8): "✅ Fully supported",
        (3, 9): "✅ Fully supported", 
        (3, 10): "✅ Fully supported",
        (3, 11): "⚠️  Limited support (some versions)",
        (3, 12): "❌ Not supported by older MediaPipe versions"
    }
    
    key = (version.major, version.minor)
    status = compatible_versions.get(key, "❓ Unknown compatibility")
    print(f"MediaPipe Compatibility: {status}")
    
    if version.minor >= 12:
        print("💡 Recommendation: Use Python 3.8-3.11 for best MediaPipe compatibility")
        return False
    elif version.minor == 11:
        print("💡 May need specific MediaPipe version")
        return True
    else:
        print("✅ Python version should work fine")
        return True

def test_mediapipe_versions():
    """Test different MediaPipe versions"""
    print("\n🖐️  MEDIAPIPE VERSION TEST")
    print("-" * 30)
    
    # List of MediaPipe versions to try (newest to oldest)
    versions_to_try = [
        "0.10.8",
        "0.10.7", 
        "0.10.5",
        "0.10.3",
        "0.10.1",
        "0.9.3.0",
        "0.9.1.0",
        "0.8.11"
    ]
    
    # Check if MediaPipe is already installed
    try:
        import mediapipe as mp
        print(f"✅ MediaPipe already installed: {mp.__version__}")
        
        # Test if it works
        try:
            hands = mp.solutions.hands.Hands()
            print("✅ MediaPipe hands module working")
            return True
        except Exception as e:
            print(f"❌ MediaPipe hands module error: {e}")
            print("🔧 MediaPipe installed but not working properly")
    except ImportError:
        print("❌ MediaPipe not found")
    
    print("\n🔧 Trying to install compatible MediaPipe version...")
    
    for version in versions_to_try:
        print(f"\n📦 Trying MediaPipe {version}...")
        try:
            # Uninstall existing version first
            subprocess.run([sys.executable, "-m", "pip", "uninstall", "mediapipe", "-y"], 
                          capture_output=True, check=False)
            
            # Install specific version
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", f"mediapipe=={version}"
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                # Test if it works
                try:
                    import mediapipe as mp
                    hands = mp.solutions.hands.Hands()
                    print(f"✅ MediaPipe {version} installed and working!")
                    return True
                except Exception as e:
                    print(f"❌ MediaPipe {version} installed but not working: {e}")
            else:
                print(f"❌ Failed to install MediaPipe {version}")
                print(f"Error: {result.stderr}")
        
        except Exception as e:
            print(f"❌ Error installing MediaPipe {version}: {e}")
    
    print("❌ Could not install any compatible MediaPipe version")
    return False

def suggest_python_installation():
    """Suggest Python installation options"""
    print("\n💡 PYTHON INSTALLATION SUGGESTIONS")
    print("-" * 40)
    
    current_version = sys.version_info
    
    if current_version.minor >= 12:
        print("🔧 Your Python version is too new for MediaPipe.")
        print("   Recommended solutions:")
        print("   1. Install Python 3.10 or 3.11 alongside current version")
        print("   2. Use pyenv to manage multiple Python versions")
        print("   3. Create virtual environment with older Python")
        
        print("\n📋 Installation commands:")
        if platform.system() == "Windows":
            print("   Windows:")
            print("   • Download Python 3.10 from python.org")
            print("   • Or use: winget install Python.Python.3.10")
            print("   • Or use Anaconda: conda create -n airwriting python=3.10")
        else:
            print("   Linux/Mac:")
            print("   • pyenv install 3.10.12")
            print("   • pyenv virtualenv 3.10.12 airwriting")
            print("   • pyenv activate airwriting")
    
    elif current_version.minor == 11:
        print("🔧 Python 3.11 should work with specific MediaPipe versions.")
        print("   Try: pip install mediapipe==0.10.8")
    
    else:
        print("✅ Your Python version should work fine with MediaPipe.")

def create_fallback_tracker():
    """Create a fallback hand tracker without MediaPipe"""
    print("\n🔧 CREATING FALLBACK HAND TRACKER")
    print("-" * 35)
    
    fallback_code = '''#!/usr/bin/env python3
"""
Fallback Hand Tracker - Works without MediaPipe
Uses basic computer vision for hand detection
"""

import cv2
import numpy as np
from collections import deque
import time

class FallbackHandTracker:
    def __init__(self):
        self.trail = deque(maxlen=100)
        self.last_position = None
        self.velocity = 0.0
        print("⚠️  Using fallback hand tracker (no MediaPipe)")
        print("   This uses basic color detection - may be less accurate")
    
    def detect_hand_color(self, frame):
        """Detect hand using skin color detection"""
        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Define skin color range (adjust as needed)
        lower_skin = np.array([0, 20, 70])
        upper_skin = np.array([20, 255, 255])
        
        # Create mask
        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
        # Apply morphological operations
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        return mask
    
    def find_fingertip(self, mask, frame):
        """Find fingertip from mask"""
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Get largest contour (hand)
        largest_contour = max(contours, key=cv2.contourArea)
        
        if cv2.contourArea(largest_contour) < 1000:
            return None
        
        # Find topmost point (fingertip)
        topmost = tuple(largest_contour[largest_contour[:, :, 1].argmin()][0])
        
        # Draw contour for visualization
        cv2.drawContours(frame, [largest_contour], -1, (0, 255, 0), 2)
        
        return topmost
    
    def process_frame(self, frame):
        """Process frame and return fingertip position"""
        # Detect hand
        mask = self.detect_hand_color(frame)
        fingertip = self.find_fingertip(mask, frame)
        
        is_writing = False
        
        if fingertip:
            # Calculate velocity
            if self.last_position:
                dx = fingertip[0] - self.last_position[0]
                dy = fingertip[1] - self.last_position[1]
                self.velocity = np.sqrt(dx*dx + dy*dy)
                is_writing = self.velocity > 5.0
            
            self.last_position = fingertip
            self.trail.append(fingertip)
            
            # Draw fingertip
            color = (0, 0, 255) if is_writing else (0, 255, 0)
            cv2.circle(frame, fingertip, 10, color, -1)
            
            # Draw trail
            if len(self.trail) > 1:
                for i in range(1, len(self.trail)):
                    cv2.line(frame, self.trail[i-1], self.trail[i], (255, 0, 0), 2)
        
        else:
            self.velocity = 0.0
        
        # Draw instructions
        cv2.putText(frame, "Fallback Tracker - Adjust lighting", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Velocity: {self.velocity:.1f}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return fingertip, self.velocity, is_writing
    
    def clear_trail(self):
        self.trail.clear()

def test_fallback_tracker():
    """Test the fallback tracker"""
    print("🧪 Testing fallback hand tracker...")
    
    tracker = FallbackHandTracker()
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Camera not available")
        return
    
    print("📋 Instructions:")
    print("   • Hold your hand up against a contrasting background")
    print("   • Adjust lighting for better detection")
    print("   • Move your hand to see tracking")
    print("   • Press ESC to exit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        fingertip, velocity, is_writing = tracker.process_frame(frame)
        
        cv2.imshow("Fallback Hand Tracker", frame)
        
        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_fallback_tracker()
'''
    
    with open("fallback_hand_tracker.py", "w") as f:
        f.write(fallback_code)
    
    print("✅ Created fallback_hand_tracker.py")
    print("   This works without MediaPipe using basic computer vision")

def main():
    """Main environment check"""
    print("🔍 AIR WRITING SYSTEM - ENVIRONMENT CHECK")
    print("=" * 60)
    
    # System info
    print_system_info()
    
    # Python compatibility
    python_ok = check_python_compatibility()
    
    if not python_ok:
        suggest_python_installation()
        return
    
    # MediaPipe test
    mediapipe_ok = test_mediapipe_versions()
    
    if not mediapipe_ok:
        print("\n🔧 MediaPipe installation failed.")
        print("Creating fallback solution...")
        create_fallback_tracker()
        
        print("\n💡 NEXT STEPS:")
        print("1. Try the fallback tracker: python fallback_hand_tracker.py")
        print("2. Or install compatible Python version (3.8-3.10)")
        print("3. Or try manual MediaPipe installation:")
        print("   pip install mediapipe --no-deps")
        print("   pip install opencv-python numpy")
    else:
        print("\n✅ ENVIRONMENT CHECK PASSED!")
        print("🎉 Your system should work with the air writing system")
        print("\n🚀 Next steps:")
        print("1. Test hand tracking: python improved_hand_tracker.py")
        print("2. Run quick setup: python quick_setup.py")
        print("3. Try simple air writing: python simple_word_airwriting.py")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Environment check interrupted")
    except Exception as e:
        print(f"\n❌ Error during environment check: {e}")
        import traceback
        traceback.print_exc()