#!/usr/bin/env python3
"""
Complete Air Writing Project Launcher
Integrates all existing components and provides multiple launch options
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

class ProjectLauncher:
    """Complete project launcher with all options"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.available_systems = {
            'integrated': {
                'file': 'air_writing_system.py',
                'description': 'Complete integrated system using all existing components',
                'features': ['Hand tracking', 'Letter recognition', 'Word correction', 'Voice feedback', 'Visual effects']
            },
            'mediapipe': {
                'file': 'mediapipe_hand_tracker.py',
                'description': 'Enhanced MediaPipe-based system with advanced features',
                'features': ['Advanced hand tracking', 'Gesture recognition', 'Multiple color schemes', 'Performance optimization']
            },
            'complete': {
                'file': 'complete_airwriting_system.py',
                'description': 'Universal system with fallback support',
                'features': ['Universal compatibility', 'Fallback tracking', 'Ensemble models', 'Multi-engine TTS']
            },
            'enhanced': {
                'file': 'enhanced_realtime_system.py',
                'description': 'Enhanced real-time system with supreme accuracy',
                'features': ['Ensemble models', 'Test-time augmentation', 'Advanced word prediction', 'Confidence filtering']
            }
        }
        
        self.utilities = {
            'setup': {
                'file': 'setup.py',
                'description': 'Setup and verify installation'
            },
            'train': {
                'file': 'training/train_optimized_accurate_model.py',
                'description': 'Train optimized accurate model'
            },
            'demo': {
                'file': 'demo_enhanced_airwriting.py',
                'description': 'Quick demo of enhanced features'
            }
        }
    
    def print_banner(self):
        """Print project banner"""
        print("\n" + "="*80)
        print("🖐️  COMPLETE AIR WRITING RECOGNITION PROJECT")
        print("="*80)
        print("🎯 Integrated solution using all existing components")
        print("📁 Project structure analyzed and optimized")
        print("🚀 Multiple launch options available")
        print("="*80 + "\n")
    
    def list_available_systems(self):
        """List all available systems"""
        print("📋 Available Air Writing Systems:")
        print("-" * 50)
        
        for key, system in self.available_systems.items():
            status = "✅" if self.check_file_exists(system['file']) else "❌"
            print(f"{status} {key.upper():<12} - {system['description']}")
            
            if self.check_file_exists(system['file']):
                print(f"   📁 File: {system['file']}")
                print(f"   🎯 Features: {', '.join(system['features'])}")
            else:
                print(f"   ⚠️  File not found: {system['file']}")
            print()
    
    def list_utilities(self):
        """List available utilities"""
        print("🔧 Available Utilities:")
        print("-" * 30)
        
        for key, util in self.utilities.items():
            status = "✅" if self.check_file_exists(util['file']) else "❌"
            print(f"{status} {key.upper():<8} - {util['description']}")
            if self.check_file_exists(util['file']):
                print(f"   📁 File: {util['file']}")
            print()
    
    def check_file_exists(self, file_path):
        """Check if file exists"""
        return (self.project_root / file_path).exists()
    
    def check_dependencies(self):
        """Check if dependencies are available"""
        print("🔍 Checking Dependencies:")
        print("-" * 30)
        
        dependencies = [
            ('opencv-python', 'cv2'),
            ('mediapipe', 'mediapipe'),
            ('numpy', 'numpy'),
            ('tensorflow', 'tensorflow'),
            ('textblob', 'textblob'),
            ('pyttsx3', 'pyttsx3'),
            ('scipy', 'scipy')
        ]
        
        available = []
        missing = []
        
        for package, import_name in dependencies:
            try:
                __import__(import_name)
                print(f"✅ {package}")
                available.append(package)
            except ImportError:
                print(f"❌ {package}")
                missing.append(package)
        
        print(f"\n📊 Summary: {len(available)}/{len(dependencies)} dependencies available")
        
        if missing:
            print(f"\n⚠️  Missing dependencies: {', '.join(missing)}")
            print("💡 Install with: pip install " + " ".join(missing))
        
        return len(missing) == 0
    
    def check_models(self):
        """Check available models"""
        print("\n🧠 Available Models:")
        print("-" * 25)
        
        models_dir = self.project_root / "models"
        if not models_dir.exists():
            print("❌ Models directory not found")
            return []
        
        model_files = list(models_dir.glob("*.h5"))
        
        if not model_files:
            print("❌ No trained models found")
            print("💡 Train a model using: python training/train_optimized_accurate_model.py")
            return []
        
        for model_file in model_files:
            file_size = model_file.stat().st_size / (1024 * 1024)  # MB
            print(f"✅ {model_file.name} ({file_size:.1f} MB)")
        
        return model_files
    
    def launch_system(self, system_name, args=None):
        """Launch specified system"""
        if system_name not in self.available_systems:
            print(f"❌ Unknown system: {system_name}")
            return False
        
        system = self.available_systems[system_name]
        file_path = self.project_root / system['file']
        
        if not file_path.exists():
            print(f"❌ File not found: {system['file']}")
            return False
        
        print(f"🚀 Launching {system_name.upper()} system...")
        print(f"📁 File: {system['file']}")
        print(f"🎯 Features: {', '.join(system['features'])}")
        
        # Build command
        cmd = [sys.executable, str(file_path)]
        if args:
            cmd.extend(args)
        
        try:
            # Launch in same environment
            subprocess.run(cmd, cwd=self.project_root)
            return True
        except KeyboardInterrupt:
            print("\n👋 System interrupted by user")
            return True
        except Exception as e:
            print(f"❌ Launch error: {e}")
            return False
    
    def launch_utility(self, util_name, args=None):
        """Launch specified utility"""
        if util_name not in self.utilities:
            print(f"❌ Unknown utility: {util_name}")
            return False
        
        util = self.utilities[util_name]
        file_path = self.project_root / util['file']
        
        if not file_path.exists():
            print(f"❌ File not found: {util['file']}")
            return False
        
        print(f"🔧 Running {util_name.upper()} utility...")
        print(f"📁 File: {util['file']}")
        
        # Build command
        cmd = [sys.executable, str(file_path)]
        if args:
            cmd.extend(args)
        
        try:
            subprocess.run(cmd, cwd=self.project_root)
            return True
        except KeyboardInterrupt:
            print("\n👋 Utility interrupted by user")
            return True
        except Exception as e:
            print(f"❌ Utility error: {e}")
            return False
    
    def interactive_launcher(self):
        """Interactive launcher menu"""
        while True:
            print("\n" + "="*60)
            print("🎮 INTERACTIVE LAUNCHER")
            print("="*60)
            print("1. Launch Air Writing System")
            print("2. Run Utilities")
            print("3. Check System Status")
            print("4. List Available Components")
            print("5. Exit")
            print("-" * 60)
            
            try:
                choice = input("Select option (1-5): ").strip()
                
                if choice == '1':
                    self.interactive_system_launcher()
                elif choice == '2':
                    self.interactive_utility_launcher()
                elif choice == '3':
                    self.check_system_status()
                elif choice == '4':
                    self.list_all_components()
                elif choice == '5':
                    print("👋 Goodbye!")
                    break
                else:
                    print("❌ Invalid choice. Please select 1-5.")
                    
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
    
    def interactive_system_launcher(self):
        """Interactive system launcher"""
        print("\n🚀 Available Air Writing Systems:")
        print("-" * 40)
        
        available_systems = []
        for i, (key, system) in enumerate(self.available_systems.items(), 1):
            if self.check_file_exists(system['file']):
                print(f"{i}. {key.upper()} - {system['description']}")
                available_systems.append(key)
            else:
                print(f"❌ {key.upper()} - File not found")
        
        if not available_systems:
            print("❌ No systems available")
            return
        
        try:
            choice = input(f"\nSelect system (1-{len(available_systems)}): ").strip()
            choice_idx = int(choice) - 1
            
            if 0 <= choice_idx < len(available_systems):
                system_name = available_systems[choice_idx]
                
                # Ask for additional arguments
                args_input = input("Additional arguments (optional): ").strip()
                args = args_input.split() if args_input else None
                
                self.launch_system(system_name, args)
            else:
                print("❌ Invalid choice")
                
        except ValueError:
            print("❌ Invalid input")
        except KeyboardInterrupt:
            print("\n👋 Cancelled")
    
    def interactive_utility_launcher(self):
        """Interactive utility launcher"""
        print("\n🔧 Available Utilities:")
        print("-" * 30)
        
        available_utils = []
        for i, (key, util) in enumerate(self.utilities.items(), 1):
            if self.check_file_exists(util['file']):
                print(f"{i}. {key.upper()} - {util['description']}")
                available_utils.append(key)
            else:
                print(f"❌ {key.upper()} - File not found")
        
        if not available_utils:
            print("❌ No utilities available")
            return
        
        try:
            choice = input(f"\nSelect utility (1-{len(available_utils)}): ").strip()
            choice_idx = int(choice) - 1
            
            if 0 <= choice_idx < len(available_utils):
                util_name = available_utils[choice_idx]
                
                # Ask for additional arguments
                args_input = input("Additional arguments (optional): ").strip()
                args = args_input.split() if args_input else None
                
                self.launch_utility(util_name, args)
            else:
                print("❌ Invalid choice")
                
        except ValueError:
            print("❌ Invalid input")
        except KeyboardInterrupt:
            print("\n👋 Cancelled")
    
    def check_system_status(self):
        """Check complete system status"""
        print("\n🔍 SYSTEM STATUS CHECK")
        print("="*50)
        
        # Check dependencies
        deps_ok = self.check_dependencies()
        
        # Check models
        models = self.check_models()
        
        # Check files
        print("\n📁 File Status:")
        print("-" * 20)
        
        all_files = {}
        all_files.update(self.available_systems)
        all_files.update(self.utilities)
        
        files_ok = 0
        total_files = len(all_files)
        
        for key, info in all_files.items():
            if self.check_file_exists(info['file']):
                print(f"✅ {info['file']}")
                files_ok += 1
            else:
                print(f"❌ {info['file']}")
        
        # Overall status
        print(f"\n📊 OVERALL STATUS")
        print("="*30)
        print(f"Dependencies: {'✅' if deps_ok else '❌'}")
        print(f"Models: {'✅' if models else '❌'} ({len(models)} available)")
        print(f"Files: {'✅' if files_ok == total_files else '❌'} ({files_ok}/{total_files})")
        
        if deps_ok and models and files_ok == total_files:
            print("\n🎉 System is ready to use!")
        else:
            print("\n⚠️  System needs attention")
            if not deps_ok:
                print("   • Install missing dependencies")
            if not models:
                print("   • Train or download models")
            if files_ok != total_files:
                print("   • Check missing files")
    
    def list_all_components(self):
        """List all project components"""
        print("\n📋 PROJECT COMPONENTS")
        print("="*50)
        
        self.list_available_systems()
        self.list_utilities()
        
        # List additional components
        print("📁 Additional Components:")
        print("-" * 30)
        
        components = [
            ("utils/", "Utility modules (hand tracking, preprocessing)"),
            ("training/", "Model training scripts"),
            ("models/", "Trained models and dictionaries"),
            ("legacy/", "Legacy implementations and demos"),
            ("demos/", "Demo applications")
        ]
        
        for path, description in components:
            full_path = self.project_root / path
            status = "✅" if full_path.exists() else "❌"
            print(f"{status} {path:<15} - {description}")

def main():
    """Main launcher function"""
    parser = argparse.ArgumentParser(description="Complete Air Writing Project Launcher")
    parser.add_argument('--system', choices=['integrated', 'mediapipe', 'complete', 'enhanced'],
                       help='Launch specific system directly')
    parser.add_argument('--utility', choices=['setup', 'train', 'demo'],
                       help='Run specific utility directly')
    parser.add_argument('--list', action='store_true',
                       help='List available components')
    parser.add_argument('--status', action='store_true',
                       help='Check system status')
    parser.add_argument('--interactive', action='store_true',
                       help='Launch interactive menu')
    parser.add_argument('args', nargs='*',
                       help='Additional arguments to pass')
    
    args = parser.parse_args()
    
    launcher = ProjectLauncher()
    launcher.print_banner()
    
    # Handle direct commands
    if args.system:
        launcher.launch_system(args.system, args.args)
    elif args.utility:
        launcher.launch_utility(args.utility, args.args)
    elif args.list:
        launcher.list_all_components()
    elif args.status:
        launcher.check_system_status()
    elif args.interactive or len(sys.argv) == 1:
        # Default to interactive if no args or explicit request
        launcher.interactive_launcher()
    else:
        # Show help
        parser.print_help()

if __name__ == "__main__":
    main()