#!/usr/bin/env python3
"""
MediaPipe Fix Script
Automatically finds and installs compatible MediaPipe version
"""

import sys
import subprocess
import platform

def get_python_version():
    """Get current Python version"""
    return sys.version_info

def get_compatible_mediapipe_versions():
    """Get list of MediaPipe versions compatible with current Python"""
    python_version = get_python_version()
    
    # MediaPipe compatibility matrix
    if python_version.minor == 8:
        return [
            "0.10.8", "0.10.7", "0.10.5", "0.10.3", "0.10.1",
            "0.9.3.0", "0.9.1.0", "0.8.11", "0.8.10"
        ]
    elif python_version.minor == 9:
        return [
            "0.10.8", "0.10.7", "0.10.5", "0.10.3", "0.10.1",
            "0.9.3.0", "0.9.1.0"
        ]
    elif python_version.minor == 10:
        return [
            "0.10.8", "0.10.7", "0.10.5", "0.10.3", "0.10.1"
        ]
    elif python_version.minor == 11:
        return [
            "0.10.8", "0.10.7", "0.10.5"
        ]
    else:
        # Python 3.12+ - very limited support
        return ["0.10.8"]

def uninstall_mediapipe():
    """Uninstall existing MediaPipe"""
    print("🗑️  Uninstalling existing MediaPipe...")
    try:
        subprocess.run([
            sys.executable, "-m", "pip", "uninstall", "mediapipe", "-y"
        ], capture_output=True, check=False)
        print("✅ Existing MediaPipe uninstalled")
    except Exception as e:
        print(f"⚠️  Could not uninstall MediaPipe: {e}")

def install_mediapipe_version(version):
    """Install specific MediaPipe version"""
    print(f"📦 Installing MediaPipe {version}...")
    
    try:
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", f"mediapipe=={version}"
        ], capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print(f"✅ MediaPipe {version} installed successfully")
            return True
        else:
            print(f"❌ Failed to install MediaPipe {version}")
            if result.stderr:
                print(f"Error: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"⏰ Installation timeout for MediaPipe {version}")
        return False
    except Exception as e:
        print(f"❌ Error installing MediaPipe {version}: {e}")
        return False

def test_mediapipe():
    """Test if MediaPipe works"""
    try:
        import mediapipe as mp
        
        # Test basic import
        hands = mp.solutions.hands
        drawing = mp.solutions.drawing_utils
        
        # Test hands initialization
        hands_detector = hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        print(f"✅ MediaPipe {mp.__version__} is working correctly")
        return True
        
    except ImportError as e:
        print(f"❌ MediaPipe import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ MediaPipe test failed: {e}")
        return False

def install_dependencies():
    """Install required dependencies"""
    print("📦 Installing dependencies...")
    
    dependencies = [
        "opencv-python==4.8.1.78",
        "numpy>=1.21.0",
        "protobuf>=3.11,<4"
    ]
    
    for dep in dependencies:
        try:
            subprocess.run([
                sys.executable, "-m", "pip", "install", dep
            ], capture_output=True, check=True)
            print(f"✅ Installed {dep}")
        except subprocess.CalledProcessError:
            print(f"⚠️  Could not install {dep}")

def try_alternative_installation():
    """Try alternative MediaPipe installation methods"""
    print("\n🔧 Trying alternative installation methods...")
    
    methods = [
        # Method 1: Install with no dependencies
        {
            "name": "No dependencies",
            "command": [sys.executable, "-m", "pip", "install", "mediapipe", "--no-deps"]
        },
        # Method 2: Install from specific index
        {
            "name": "PyPI index",
            "command": [sys.executable, "-m", "pip", "install", "mediapipe", "--index-url", "https://pypi.org/simple/"]
        },
        # Method 3: Force reinstall
        {
            "name": "Force reinstall",
            "command": [sys.executable, "-m", "pip", "install", "mediapipe", "--force-reinstall", "--no-cache-dir"]
        }
    ]
    
    for method in methods:
        print(f"\n📦 Trying: {method['name']}")
        try:
            result = subprocess.run(method["command"], capture_output=True, text=True, timeout=300)
            if result.returncode == 0:
                print(f"✅ {method['name']} succeeded")
                if test_mediapipe():
                    return True
            else:
                print(f"❌ {method['name']} failed")
        except Exception as e:
            print(f"❌ {method['name']} error: {e}")
    
    return False

def main():
    """Main MediaPipe fix function"""
    print("🔧 MEDIAPIPE FIX SCRIPT")
    print("=" * 40)
    
    python_version = get_python_version()
    print(f"Python Version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # Check if MediaPipe already works
    print("\n🧪 Testing current MediaPipe installation...")
    if test_mediapipe():
        print("🎉 MediaPipe is already working! No fix needed.")
        return True
    
    # Install dependencies first
    install_dependencies()
    
    # Get compatible versions
    compatible_versions = get_compatible_mediapipe_versions()
    print(f"\n📋 Compatible MediaPipe versions for Python {python_version.minor}: {len(compatible_versions)}")
    
    if not compatible_versions:
        print("❌ No compatible MediaPipe versions found for your Python version")
        print("💡 Consider using Python 3.8-3.10 for best compatibility")
        return False
    
    # Uninstall existing version
    uninstall_mediapipe()
    
    # Try each compatible version
    for version in compatible_versions:
        print(f"\n🎯 Trying MediaPipe {version}...")
        
        if install_mediapipe_version(version):
            if test_mediapipe():
                print(f"🎉 SUCCESS! MediaPipe {version} is working")
                return True
            else:
                print(f"⚠️  MediaPipe {version} installed but not working")
        
        # Uninstall failed version
        uninstall_mediapipe()
    
    # Try alternative methods
    print("\n🔄 Standard installation failed, trying alternatives...")
    if try_alternative_installation():
        print("🎉 Alternative installation succeeded!")
        return True
    
    # Final failure
    print("\n❌ MEDIAPIPE INSTALLATION FAILED")
    print("💡 Suggestions:")
    print("   1. Use Python 3.8-3.10 for best compatibility")
    print("   2. Try manual installation:")
    print("      pip install mediapipe --no-deps")
    print("      pip install opencv-python numpy protobuf")
    print("   3. Use the fallback hand tracker (no MediaPipe required)")
    print("   4. Check your internet connection and try again")
    
    return False

if __name__ == "__main__":
    try:
        success = main()
        if success:
            print("\n✅ MediaPipe fix completed successfully!")
            print("🚀 You can now run: python improved_hand_tracker.py")
        else:
            print("\n❌ MediaPipe fix failed")
            print("🔧 Try running: python check_environment.py")
    except KeyboardInterrupt:
        print("\n👋 MediaPipe fix interrupted")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()