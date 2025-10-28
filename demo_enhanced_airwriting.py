#!/usr/bin/env python3
"""
Enhanced Air Writing Demo
Quick demo of the enhanced air writing system with all features
"""

import sys
import os

def main():
    """Run the enhanced air writing demo"""
    print("🚀 ENHANCED AIR WRITING DEMO")
    print("=" * 50)
    
    try:
        # Import the enhanced system
        from mediapipe_hand_tracker import CompleteAirWritingSystem
        
        print("✅ Enhanced system loaded successfully")
        print("\n🎯 Features included:")
        print("   • Real-time finger tracking (30+ FPS)")
        print("   • Smooth, glowing, animated trails")
        print("   • Smart word recognition & auto-correction")
        print("   • Voice feedback with multiple TTS engines")
        print("   • Gesture-based controls (open hand to clear)")
        print("   • Multiple color schemes and visual effects")
        print("   • Background blur and performance optimization")
        print("   • Comprehensive keyboard shortcuts")
        print("   • Session logging to output_log.txt")
        
        print("\n🖐️ Instructions:")
        print("   1. Hold your INDEX finger up (other fingers curled)")
        print("   2. Write letters in the air slowly and clearly")
        print("   3. Pause briefly between letters")
        print("   4. Pause longer between words (1-2 seconds)")
        print("   5. Use OPEN HAND gesture to clear canvas")
        
        print("\n⌨️ Quick Controls:")
        print("   SPACE - Complete letter    C - Clear word")
        print("   1-4   - Color schemes      T - Toggle trail")
        print("   ESC   - Exit system")
        
        input("\nPress ENTER to start the enhanced air writing system...")
        
        # Initialize and run the system
        system = CompleteAirWritingSystem()
        system.run()
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("\n🔧 Please ensure all dependencies are installed:")
        print("   pip install -r requirements.txt")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n🔧 Troubleshooting:")
        print("   • Check camera permissions")
        print("   • Ensure good lighting")
        print("   • Run setup: python setup.py")
        
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()