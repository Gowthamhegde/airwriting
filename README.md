# 🖐️ Real-Time Air Writing Recognition System

A complete real-time air writing recognition system that detects hand movements to write words in the air using computer vision and machine learning.

## ✨ Features

- **Real-time hand tracking** using MediaPipe
- **Gesture control** (open/closed hand detection)
- **Letter recognition** with machine learning
- **Word auto-correction** using NLP
- **Voice feedback** with text-to-speech
- **Smooth trajectory tracking** with filtering
- **Multiple system variants** for different use cases

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/Gowthamhegde/airwriting.git
cd airwriting
```

### 2. Set Up Virtual Environment
```bash
# Create virtual environment
python -m venv airwriting-env

# Activate virtual environment
# On Windows:
airwriting-env\Scripts\activate
# On macOS/Linux:
source airwriting-env/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the System
```bash
# Main working system
python air_writing_system.py

# Or use the launcher
python launch_airwriting_system.py
```

## 📋 System Requirements

- **Python 3.8+** (Note: MediaPipe may not work with Python 3.13+)
- **Webcam** for hand tracking
- **Windows/macOS/Linux** support
- **4GB+ RAM** recommended

## 🎯 How to Use

1. **Start the system** - Run one of the Python scripts
2. **Show your hand** to the camera
3. **Open hand** = Start tracking (green trail appears)
4. **Write letters** in the air with your index finger
5. **Close hand** = Stop tracking and recognize word
6. **Listen** to the spoken word result

### Controls
- **Open hand** - Start/continue tracking
- **Closed hand** - Stop and recognize word
- **C key** - Clear canvas
- **ESC key** - Exit system

## 📁 Project Structure

```
airwriting/
├── air_writing_system.py          # Main working system
├── complete_realtime_airwriting.py # Enhanced version
├── ultimate_airwriting_system.py   # Advanced features
├── enhanced_realtime_airwriting.py # Real-time optimized
├── launch_airwriting_system.py     # System launcher
├── modules/                        # Core modules
│   ├── ensemble_letter_recognition.py
│   └── advanced_hand_detection.py
├── models/                         # ML models (not in repo)
├── training/                       # Training scripts
├── utils/                          # Utility functions
└── requirements.txt               # Dependencies
```

## 🔧 Available Systems

1. **`air_writing_system.py`** - Main stable system (recommended)
2. **`complete_realtime_airwriting.py`** - Enhanced with advanced features
3. **`ultimate_airwriting_system.py`** - Most advanced with gesture controls
4. **`enhanced_realtime_airwriting.py`** - Real-time optimized version

## 🛠️ Setup Scripts

- **`setup_complete_system.py`** - Comprehensive setup with tests
- **`launch_airwriting_system.py`** - Easy launcher script

## 📦 Dependencies

- **opencv-python** - Computer vision
- **mediapipe** - Hand tracking
- **numpy** - Numerical computing
- **tensorflow** - Machine learning
- **scikit-learn** - ML algorithms
- **pyttsx3** - Text-to-speech
- **textblob** - NLP and auto-correction

## 🎯 Supported Words

The system recognizes common 3-letter words including:
- CAT, DOG, BAT, RAT, HAT, MAT, SAT, FAT, PAT
- BIG, RED, YES, NO, TOP, BOX, CAR, RUN, SUN
- And many more...

## 🔍 Troubleshooting

### Common Issues

1. **MediaPipe not found**
   - Use Python 3.8-3.11 (MediaPipe doesn't support 3.13+ yet)
   - Install in virtual environment

2. **Camera not working**
   - Check camera permissions
   - Try different camera index in code
   - Ensure no other apps are using camera

3. **No voice feedback**
   - Check audio drivers
   - Try different TTS engine settings
   - System will work without audio

4. **Poor recognition**
   - Ensure good lighting
   - Write letters slowly and clearly
   - Keep hand steady during writing

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is open source. Feel free to use and modify for educational and research purposes.

## 🙏 Acknowledgments

- **MediaPipe** team for hand tracking
- **OpenCV** community for computer vision tools
- **TensorFlow** team for machine learning framework

## 📞 Support

If you encounter issues:
1. Check the troubleshooting section
2. Review system requirements
3. Open an issue on GitHub

---

**Happy Air Writing! ✍️**