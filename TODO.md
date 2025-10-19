# TODO: Improve Alphabet Tracking in AirWriting System

## Tasks
- [x] Fix train_cnn.py to use EMNIST letters dataset instead of MNIST digits
- [x] Retrain the CNN model with correct dataset
- [ ] Update app.py to load the new model (no changes needed, same filename)
- [ ] Tune letter segmentation parameters in app.py for better tracking
- [ ] Test the improved system

## Details
- Current issue: Model trained on MNIST digits (0-9) filtered incorrectly for alphabets
- Solution: Use EMNIST letters dataset (A-Z) for training
- Parameters to tune: LETTER_PAUSE_THRESHOLD, WORD_PAUSE_THRESHOLD, VELOCITY_THRESHOLD
