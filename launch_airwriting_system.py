#!/usr/bin/env python3
"""
Simple launcher for the Air Writing System
Activates virtual environment and runs the system
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    """Launch the air writing system"""
    print("🖐️ AIR WRITING SYSTEM LAUNCHER")
    print("=" * 40)
    
    # Check if we're in the project directory
    project_dir = Path(__file__).parent
    os.chdir(project_dir)
    
    # Check if virtual environment exists
    venv_path = project_dir / "airwriting-env"
    if not venv_path.exists():
        print("❌ Virtual environment not found!")
        print("Please run setup first or activate the environment manually")
        return
    
    # Check if main system file exists
    system_file = project_dir / "air_writing_system.py"
    if not system_file.exists():
        print("❌ Air writing system file not found!")
        return
    
    print("🚀 Launching Air Writing System...")
    print("\n📋 Instructions:")
    print("   • Hold up your INDEX FINGER (other fingers curled)")
    print("   • Write letters in the air slowly and clearly")
    print("   • System will auto-correct and speak recognized words")
    print("   • Press ESC to exit")
    print("\n🎯 Try words like: CAT, DOG, BAT, RAT, HAT, etc.")
    print("=" * 40)
    
    try:
        # Run the system with the virtual environment Python
        if sys.platform == "win32":
            python_exe = venv_path / "Scripts" / "python.exe"
        else:
            python_exe = venv_path / "bin" / "python"
        
        subprocess.run([str(python_exe), "air_writing_system.py"])
        
    except KeyboardInterrupt:
        print("\n👋 System interrupted by user")
    except FileNotFoundError:
        print("❌ Could not find Python executable in virtual environment")
        print("Try running: python air_writing_system.py")
    except Exception as e:
        print(f"❌ Error launching system: {e}")
    
    print("🏁 Launcher finished!")

if __name__ == "__main__":
    main()