#!/usr/bin/env python3
"""
Air Writing Recognition System Demo
Comprehensive demonstration of all system features
"""

import cv2
import numpy as np
import time
import argparse
from enhanced_airwriting_app import EnhancedAirWritingSystem
from main_airwriting_system import AirWritingSystem

class AirWritingDemo:
    def __init__(self):
        self.demo_modes = {
            'basic': 'Basic air writing with simple recognition',
            'enhanced': 'Enhanced system with advanced features',
            'comparison': 'Side-by-side comparison of both systems'
        }
    
    def print_welcome(self):
        """Print welcome message and demo options"""
        print("\n" + "="*70)
        print("🖐️  REAL-TIME AIR WRITING RECOGNITION SYSTEM DEMO")
        print("="*70)
        print("This demo showcases a complete air writing recognition system")
        print("that can recognize words using index finger gestures.")
        print("\n📋 FEATURES:")
        print("  ✅ Real-time hand tracking with MediaPipe")
        print("  ✅ CNN-based letter recognition (A-Z)")
        print("  ✅ Automatic word completion and correction")
        print("  ✅ Text-to-speech output")
        print("  ✅ Enhanced gesture recognition")
        print("  ✅ Performance statistics")
        print("  ✅ Multiple UI modes")
        print("\n🎯 DEMO MODES:")
        for mode, description in self.demo_modes.items():
            print(f"  {mode}: {description}")
        print("="*70)
    
    def run_basic_demo(self):
        """Run basic air writing demo"""
        print("\n🚀 Starting Basic Air Writing Demo...")
        print("This demonstrates the core functionality with simple UI.")
        
        try:
            system = AirWritingSystem()
            system.run()
        except Exception as e:
            print(f"Error in basic demo: {e}")
    
    def run_enhanced_demo(self):
        """Run enhanced air writing demo"""
        print("\n🚀 Starting Enhanced Air Writing Demo...")
        print("This demonstrates advanced features with enhanced UI.")
        
        try:
            system = EnhancedAirWritingSystem()
            system.run()
        except Exception as e:
            print(f"Error in enhanced demo: {e}")
    
    def run_comparison_demo(self):
        """Run comparison demo (sequential)"""
        print("\n🚀 Starting Comparison Demo...")
        print("You'll first try the basic system, then the enhanced system.")
        
        print("\n--- BASIC SYSTEM ---")
        input("Press Enter to start basic system...")
        try:
            system = AirWritingSystem()
            system.run()
        except Exception as e:
            print(f"Error in basic system: {e}")
        
        print("\n--- ENHANCED SYSTEM ---")
        input("Press Enter to start enhanced system...")
        try:
            system = EnhancedAirWritingSystem()
            system.run()
        except Exception as e:
            print(f"Error in enhanced system: {e}")
    
    def check_requirements(self):
        """Check if all requirements are met"""
        print("\n🔍 Checking system requirements...")
        
        requirements_met = True
        
        # Check camera
        try:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                print("❌ Camera not accessible")
                requirements_met = False
            else:
                print("✅ Camera accessible")
                cap.release()
        except Exception:
            print("❌ OpenCV camera error")
            requirements_met = False
        
        # Check model file
        import os
        if os.path.exists("models/letter_recognition.h5"):
            print("✅ Letter recognition model found")
        elif os.path.exists("models/letter_recognition_enhanced.h5"):
            print("✅ Enhanced letter recognition model found")
        else:
            print("❌ No trained model found")
            print("   Please run train_cnn.py or train_enhanced_model.py first")
            requirements_met = False
        
        # Check dependencies
        try:
            import mediapipe
            print("✅ MediaPipe available")
        except ImportError:
            print("❌ MediaPipe not installed")
            requirements_met = False
        
        try:
            import tensorflow
            print("✅ TensorFlow available")
        except ImportError:
            print("❌ TensorFlow not installed")
            requirements_met = False
        
        try:
            import textblob
            print("✅ TextBlob available")
        except ImportError:
            print("❌ TextBlob not installed")
            requirements_met = False
        
        try:
            import pyttsx3
            print("✅ Text-to-speech available")
        except ImportError:
            print("❌ pyttsx3 not installed")
            requirements_met = False
        
        return requirements_met
    
    def show_usage_tips(self):
        """Show usage tips for better experience"""
        print("\n💡 USAGE TIPS FOR BEST RESULTS:")
        print("="*50)
        print("📹 Camera Setup:")
        print("  • Ensure good lighting")
        print("  • Position camera at eye level")
        print("  • Keep background simple and contrasting")
        print("  • Maintain 2-3 feet distance from camera")
        
        print("\n✋ Hand Position:")
        print("  • Extend only your index finger")
        print("  • Keep other fingers curled")
        print("  • Write at a comfortable speed")
        print("  • Make clear, distinct letter shapes")
        
        print("\n✍️ Writing Technique:")
        print("  • Write letters larger than normal")
        print("  • Pause briefly between letters")
        print("  • Pause longer between words")
        print("  • Use consistent writing style")
        
        print("\n🎛️ Controls:")
        print("  • SPACE: Force end current letter")
        print("  • S: Speak current word")
        print("  • C: Clear current word")
        print("  • ESC: Exit application")
        print("="*50)
    
    def run_interactive_demo(self):
        """Run interactive demo with user choices"""
        while True:
            print("\n🎮 INTERACTIVE DEMO MENU")
            print("-" * 30)
            print("1. Basic Air Writing Demo")
            print("2. Enhanced Air Writing Demo")
            print("3. Comparison Demo")
            print("4. Show Usage Tips")
            print("5. Check System Requirements")
            print("6. Exit")
            
            try:
                choice = input("\nSelect option (1-6): ").strip()
                
                if choice == '1':
                    self.run_basic_demo()
                elif choice == '2':
                    self.run_enhanced_demo()
                elif choice == '3':
                    self.run_comparison_demo()
                elif choice == '4':
                    self.show_usage_tips()
                elif choice == '5':
                    self.check_requirements()
                elif choice == '6':
                    print("👋 Thank you for trying the Air Writing Demo!")
                    break
                else:
                    print("❌ Invalid choice. Please select 1-6.")
                    
            except KeyboardInterrupt:
                print("\n👋 Demo interrupted by user")
                break
            except Exception as e:
                print(f"❌ Error: {e}")

def main():
    """Main demo function"""
    parser = argparse.ArgumentParser(description='Air Writing Recognition System Demo')
    parser.add_argument('--mode', choices=['basic', 'enhanced', 'comparison', 'interactive'], 
                       default='interactive', help='Demo mode to run')
    parser.add_argument('--check-requirements', action='store_true', 
                       help='Check system requirements only')
    
    args = parser.parse_args()
    
    demo = AirWritingDemo()
    demo.print_welcome()
    
    if args.check_requirements:
        demo.check_requirements()
        return
    
    # Check requirements first
    if not demo.check_requirements():
        print("\n❌ System requirements not met. Please install missing dependencies.")
        return
    
    demo.show_usage_tips()
    
    # Run selected demo mode
    if args.mode == 'basic':
        demo.run_basic_demo()
    elif args.mode == 'enhanced':
        demo.run_enhanced_demo()
    elif args.mode == 'comparison':
        demo.run_comparison_demo()
    else:  # interactive
        demo.run_interactive_demo()

if __name__ == "__main__":
    main()