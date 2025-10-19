# Complete Air Writing Recognition System

A comprehensive, self-contained real-time air writing recognition system that recognizes words using index finger gestures. This integrated solution uses computer vision, machine learning, and natural language processing with fallback support for all components.

## 🌟 Features

### Core Functionality
- **Universal Hand Tracking**: MediaPipe with fallback to color-based tracking
- **Letter Recognition**: CNN-based model with demo mode fallback
- **Word Formation**: Automatic letter segmentation and word completion
- **Auto-correction**: Intelligent word correction with custom dictionary
- **Text-to-Speech**: Speaks recognized words with fallback
- **Integrated UI**: Complete interface with status indicators and controls

### Advanced Features
- **Fallback Support**: Works without MediaPipe, TensorFlow, or TTS libraries
- **Gesture Recognition**: Detects writing posture vs other hand gestures
- **Performance Monitoring**: Real-time FPS, letter/word counters
- **Word Suggestions**: Real-time completion suggestions
- **Session Summary**: Tracks and displays session statistics
- **Robust Error Handling**: Graceful degradation when components fail

## 🚀 Quick Start

### 1. Installation

#### Option A: Windows Easy Install (Recommended)
```bash
# Clone the repository
git clone <repository-url>
cd air-writing-recognition

# Run Windows installer
python install_windows.py
```

#### Option B: Manual Installation
```bash
# Install dependencies
pip install -r requirements.txt
```

### 2. Run the System

```bash
# Run the complete integrated system
python complete_airwriting_system.py
```

## 📁 Project Structure

```
air-writing-recognition/
├── complete_airwriting_system.py  # Main integrated application
├── train_advanced_model.py       # Model training scripts
├── train_enhanced_model.py       # Enhanced model training
├── train_cnn.py                  # Basic model training
├── utils/                        # Utility modules
│   ├── hand_tracker.py          # Hand tracking utilities
│   └── preprocessing.py         # Image preprocessing
├── models/                      # Trained models directory
├── legacy/                      # Previous versions and demos
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## 🎮 Usage Instructions

### Hand Position
1. **Writing Gesture**: Extend only your index finger, keep other fingers curled
2. **Distance**: Maintain 2-3 feet from the camera
3. **Lighting**: Ensure good lighting with contrasting background

### Writing Technique
1. **Letter Formation**: Write letters larger than normal handwriting
2. **Speed**: Write at a comfortable, consistent speed
3. **Pauses**: Brief pause between letters, longer pause between words
4. **Clarity**: Make distinct letter shapes for better recognition

### Controls

#### Basic Controls (All Modes)
- **SPACE**: Manually end current letter
- **S**: Speak current word
- **C**: Clear current word
- **ESC**: Exit application

#### Enhanced Mode Additional Controls
- **T**: Toggle trail visibility
- **D**: Toggle debug information
- **A**: Toggle auto-correction
- **H**: Toggle word suggestions
- **R**: Reset session statistics

## 🧠 System Architecture

### 1. Hand Tracking Module (`utils/hand_tracker.py`)
- MediaPipe-based hand detection
- Gesture recognition (writing vs non-writing poses)
- Smoothed fingertip tracking with velocity calculation
- Enhanced trail visualization

### 2. Preprocessing Module (`utils/preprocessing.py`)
- Path normalization and smoothing
- Feature extraction (aspect ratio, direction changes, etc.)
- Image enhancement for better recognition
- Letter region extraction

### 3. Recognition Models
- **Basic Model**: Simple CNN trained on EMNIST letters
- **Enhanced Model**: Deeper architecture with batch normalization and dropout
- **Input**: 28x28 grayscale images
- **Output**: 26-class probability distribution (A-Z)

### 4. Word Recognition (`word_recognition.py`)
- Custom dictionary with common words
- Letter confusion matrix for contextual correction
- Word completion suggestions
- Confidence scoring

### 5. Applications
- **Basic System**: Core functionality with simple UI
- **Enhanced System**: Advanced features with comprehensive UI
- **Demo System**: Interactive demonstration of all features

## 📊 Performance

### Model Accuracy
- **Basic Model**: ~85-90% letter accuracy
- **Enhanced Model**: ~92-95% letter accuracy
- **Word Accuracy**: ~80-85% with auto-correction

### System Requirements
- **Camera**: Any USB webcam (720p recommended)
- **CPU**: Modern multi-core processor
- **RAM**: 4GB minimum, 8GB recommended
- **Python**: 3.8 or higher

## 🔧 Configuration

### Model Parameters
```python
# In enhanced_airwriting_app.py
LETTER_PAUSE_THRESHOLD = 20      # Frames to end letter
WORD_PAUSE_THRESHOLD = 120       # Frames to end word
VELOCITY_THRESHOLD = 2.5         # Movement detection threshold
CONFIDENCE_THRESHOLD = 0.4       # Letter recognition threshold
```

### Hand Tracking Parameters
```python
# In utils/hand_tracker.py
trail_length = 150               # Trail history length
alpha = 0.2                      # Smoothing factor
min_detection_confidence = 0.8   # Hand detection threshold
```

## 🐛 Troubleshooting

### Common Issues

1. **Installation Problems (Windows)**
   - Use `python install_windows.py` for automated setup
   - If MediaPipe fails, try: `pip install mediapipe --no-deps`
   - For TensorFlow issues, try: `pip install tensorflow-cpu`
   - Run `python simple_airwriting_demo.py` to test basic functionality

2. **Camera Not Working**
   - Check camera permissions in Windows Settings
   - Ensure no other applications are using the camera
   - Try different camera indices (0, 1, 2...)
   - Run `python test_system.py` after installation

3. **Poor Recognition Accuracy**
   - Improve lighting conditions
   - Write letters larger and more clearly
   - Ensure proper hand gesture (index finger extended)
   - Check camera positioning (2-3 feet distance)

4. **Model Not Found Error**
   - Run training script first: `python train_enhanced_model.py`
   - Check if models directory exists
   - Verify model file path

5. **Slow Performance**
   - Reduce camera resolution
   - Close other applications
   - Use basic mode instead of enhanced mode
   - Try `simple_airwriting_demo.py` for lightweight version

### Debug Mode
Enable debug mode in enhanced system to see:
- FPS counter
- Velocity measurements
- Path point counts
- Gesture confidence scores

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **MediaPipe**: Google's framework for hand tracking
- **EMNIST Dataset**: Extended MNIST for letter recognition
- **TensorFlow**: Machine learning framework
- **OpenCV**: Computer vision library
- **TextBlob**: Natural language processing

## 📞 Support

For issues, questions, or contributions:
1. Check the troubleshooting section
2. Search existing issues
3. Create a new issue with detailed description
4. Include system information and error messages

---

**Happy Air Writing! ✋✍️**