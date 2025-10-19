#!/usr/bin/env python3
"""
Setup script for Air Writing Recognition System
Handles installation, model training, and system verification
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

class AirWritingSetup:
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.models_dir = self.project_root / "models"
        self.requirements_file = self.project_root / "requirements.txt"
    
    def print_banner(self):
        """Print setup banner"""
        print("\n" + "="*70)
        print("🖐️  AIR WRITING RECOGNITION SYSTEM SETUP")
        print("="*70)
        print("This script will help you set up the air writing recognition system.")
        print("="*70 + "\n")
    
    def check_python_version(self):
        """Check if Python version is compatible"""
        print("🐍 Checking Python version...")
        
        if sys.version_info < (3, 8):
            print("❌ Python 3.8 or higher is required")
            print(f"   Current version: {sys.version}")
            return False
        
        print(f"✅ Python {sys.version.split()[0]} is compatible")
        return True
    
    def install_dependencies(self, force=False):
        """Install required dependencies"""
        print("\n📦 Installing dependencies...")
        
        if not self.requirements_file.exists():
            print("❌ requirements.txt not found")
            return False
        
        try:
            cmd = [sys.executable, "-m", "pip", "install", "-r", str(self.requirements_file)]
            if force:
                cmd.append("--force-reinstall")
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ Dependencies installed successfully")
                return True
            else:
                print("❌ Failed to install dependencies")
                print(f"Error: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"❌ Error installing dependencies: {e}")
            return False
    
    def create_directories(self):
        """Create necessary directories"""
        print("\n📁 Creating directories...")
        
        directories = [
            self.models_dir,
            self.project_root / "data",
            self.project_root / "logs"
        ]
        
        for directory in directories:
            try:
                directory.mkdir(exist_ok=True)
                print(f"✅ Created/verified directory: {directory.name}")
            except Exception as e:
                print(f"❌ Failed to create directory {directory}: {e}")
                return False
        
        return True
    
    def check_camera(self):
        """Check if camera is accessible"""
        print("\n📹 Checking camera access...")
        
        try:
            import cv2
            cap = cv2.VideoCapture(0)
            
            if not cap.isOpened():
                print("❌ Camera not accessible")
                print("   Please check camera permissions and connections")
                return False
            
            # Try to read a frame
            ret, frame = cap.read()
            cap.release()
            
            if ret and frame is not None:
                print("✅ Camera is working properly")
                return True
            else:
                print("❌ Camera not providing frames")
                return False
                
        except ImportError:
            print("❌ OpenCV not installed")
            return False
        except Exception as e:
            print(f"❌ Camera check failed: {e}")
            return False
    
    def train_model(self, model_type="enhanced"):
        """Train the recognition model"""
        print(f"\n🧠 Training {model_type} model...")
        
        if model_type == "enhanced":
            script_name = "train_enhanced_model.py"
        else:
            script_name = "train_cnn.py"
        
        script_path = self.project_root / script_name
        
        if not script_path.exists():
            print(f"❌ Training script {script_name} not found")
            return False
        
        try:
            print("   This may take 10-30 minutes depending on your hardware...")
            result = subprocess.run([sys.executable, str(script_path)], 
                                  capture_output=False, text=True)
            
            if result.returncode == 0:
                print("✅ Model training completed successfully")
                return True
            else:
                print("❌ Model training failed")
                return False
                
        except Exception as e:
            print(f"❌ Error during model training: {e}")
            return False
    
    def verify_installation(self):
        """Verify that everything is set up correctly"""
        print("\n🔍 Verifying installation...")
        
        checks = []
        
        # Check model files
        model_files = [
            "letter_recognition.h5",
            "letter_recognition_enhanced.h5"
        ]
        
        model_found = False
        for model_file in model_files:
            model_path = self.models_dir / model_file
            if model_path.exists():
                print(f"✅ Found model: {model_file}")
                model_found = True
                break
        
        if not model_found:
            print("❌ No trained model found")
            checks.append(False)
        else:
            checks.append(True)
        
        # Check key dependencies
        dependencies = [
            ("cv2", "OpenCV"),
            ("mediapipe", "MediaPipe"),
            ("tensorflow", "TensorFlow"),
            ("textblob", "TextBlob"),
            ("pyttsx3", "Text-to-Speech"),
            ("numpy", "NumPy"),
            ("scipy", "SciPy")
        ]
        
        for module, name in dependencies:
            try:
                __import__(module)
                print(f"✅ {name} is available")
                checks.append(True)
            except ImportError:
                print(f"❌ {name} not found")
                checks.append(False)
        
        # Overall result
        if all(checks):
            print("\n🎉 Installation verification successful!")
            print("   You can now run the air writing system.")
            return True
        else:
            print("\n❌ Installation verification failed")
            print("   Please address the issues above.")
            return False
    
    def run_quick_test(self):
        """Run a quick test of the system"""
        print("\n🧪 Running quick system test...")
        
        try:
            # Test imports
            from utils.hand_tracker import HandTracker
            from utils.preprocessing import draw_path_on_blank
            from word_recognition import correct_word_enhanced
            
            # Test hand tracker initialization
            tracker = HandTracker()
            print("✅ Hand tracker initialized")
            
            # Test preprocessing
            test_path = [(10, 10), (20, 20), (30, 30)]
            img = draw_path_on_blank(test_path)
            print("✅ Preprocessing functions working")
            
            # Test word correction
            corrected = correct_word_enhanced("HELO")
            print(f"✅ Word correction working: HELO -> {corrected}")
            
            print("✅ Quick test passed!")
            return True
            
        except Exception as e:
            print(f"❌ Quick test failed: {e}")
            return False
    
    def interactive_setup(self):
        """Run interactive setup process"""
        print("🎮 Starting interactive setup...")
        
        steps = [
            ("Check Python version", self.check_python_version),
            ("Install dependencies", lambda: self.install_dependencies()),
            ("Create directories", self.create_directories),
            ("Check camera", self.check_camera),
        ]
        
        for step_name, step_func in steps:
            print(f"\n--- {step_name} ---")
            if not step_func():
                print(f"❌ Setup failed at: {step_name}")
                return False
        
        # Ask about model training
        print("\n--- Model Training ---")
        train_choice = input("Do you want to train the model now? (y/n): ").lower().strip()
        
        if train_choice in ['y', 'yes']:
            model_choice = input("Choose model type (basic/enhanced) [enhanced]: ").lower().strip()
            model_type = "enhanced" if model_choice in ['', 'enhanced'] else "basic"
            
            if not self.train_model(model_type):
                print("❌ Model training failed")
                return False
        else:
            print("⚠️  Skipping model training")
            print("   You can train later using: python train_enhanced_model.py")
        
        # Final verification
        if self.verify_installation():
            if self.run_quick_test():
                print("\n🎉 Setup completed successfully!")
                print("\nNext steps:")
                print("  1. Run demo: python demo_airwriting.py")
                print("  2. Or run enhanced system: python enhanced_airwriting_app.py")
                return True
        
        return False

def main():
    """Main setup function"""
    parser = argparse.ArgumentParser(description='Air Writing Recognition System Setup')
    parser.add_argument('--install-deps', action='store_true', 
                       help='Install dependencies only')
    parser.add_argument('--train-model', choices=['basic', 'enhanced'], 
                       help='Train model only')
    parser.add_argument('--verify', action='store_true', 
                       help='Verify installation only')
    parser.add_argument('--quick-test', action='store_true', 
                       help='Run quick test only')
    parser.add_argument('--force-reinstall', action='store_true', 
                       help='Force reinstall dependencies')
    
    args = parser.parse_args()
    
    setup = AirWritingSetup()
    setup.print_banner()
    
    # Handle specific actions
    if args.install_deps:
        setup.install_dependencies(force=args.force_reinstall)
    elif args.train_model:
        setup.train_model(args.train_model)
    elif args.verify:
        setup.verify_installation()
    elif args.quick_test:
        setup.run_quick_test()
    else:
        # Run interactive setup
        setup.interactive_setup()

if __name__ == "__main__":
    main()