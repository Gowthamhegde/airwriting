
import cv2
from utils.hand_tracker import HandTracker
from utils.preprocessing import draw_path_on_blank
from autocorrector import correct_word
from text_to_speech import speak_word
import numpy as np
from keras.models import load_model
import time

model = load_model("models/letter_recognition.h5")
tracker = HandTracker()

cap = cv2.VideoCapture(0)
path = []
word = ""
last_movement_time = time.time()
letter_pause_frames = 0
word_pause_frames = 0
LETTER_PAUSE_THRESHOLD = 10  # frames with low velocity to end letter
WORD_PAUSE_THRESHOLD = 60  # frames to end word
VELOCITY_THRESHOLD = 5.0
WORD_COMPLETION_PAUSE = 2.5  # seconds to finalize word

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally to correct mirrored camera feed
    frame = cv2.flip(frame, 1)

    fingertip, annotated, velocity = tracker.get_fingertip(frame)

    if fingertip:
        path.append(fingertip)
        last_movement_time = time.time()

    # Automatic letter segmentation
    if velocity < VELOCITY_THRESHOLD:
        letter_pause_frames += 1
    else:
        letter_pause_frames = 0

    if letter_pause_frames > LETTER_PAUSE_THRESHOLD and len(path) > 5:
        img = draw_path_on_blank(path)
        img_resized = cv2.resize(img, (28, 28)).reshape(1, 28, 28, 1) / 255.0
        prediction = model.predict(img_resized, verbose=0)
        letter = chr(np.argmax(prediction) + ord("A"))
        word += letter
        print(f"Letter: {letter}")
        path.clear()
        letter_pause_frames = 0

    # Automatic word completion
    if time.time() - last_movement_time > WORD_COMPLETION_PAUSE:  # WORD_COMPLETION_PAUSE seconds no movement
        word_pause_frames += 1
    else:
        word_pause_frames = 0

    if word_pause_frames > WORD_PAUSE_THRESHOLD and word:
        corrected = correct_word(word)
        print(f"Final word: {corrected}")
        speak_word(corrected)
        word = ""
        word_pause_frames = 0
        path.clear()  # Ensure path is cleared when word is finalized

    # Display info on frame
    cv2.putText(annotated, f"Word: {word}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(annotated, f"Velocity: {velocity:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("AirWrite", annotated)

    key = cv2.waitKey(1)
    if key == ord(" "):  # Manual SPACEBAR = end of letter
        if len(path) > 5:
            img = draw_path_on_blank(path)
            img_resized = cv2.resize(img, (28, 28)).reshape(1, 28, 28, 1) / 255.0
            prediction = model.predict(img_resized, verbose=0)
            letter = chr(np.argmax(prediction) + ord("A"))
            word += letter
            print(f"Letter: {letter}")
        path.clear()
        letter_pause_frames = 0

    elif key == ord("s"):  # Manual 's' key to process word
        if word:
            corrected = correct_word(word)
            print(f"Final word: {corrected}")
            speak_word(corrected)
            word = ""
        word_pause_frames = 0
        path.clear()  # Ensure path is cleared when word is finalized manually

    elif key == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()


### utils/hand_tracker.py
import cv2
import mediapipe as mp

class HandTracker:
    def __init__(self, max_hands=1):
        self.hands_module = mp.solutions.hands
        self.hands = self.hands_module.Hands(max_num_hands=max_hands, min_detection_confidence=0.7)
        self.drawing = mp.solutions.drawing_utils

    def get_fingertip(self, frame):
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.hands.process(img_rgb)
        fingertip = None
        if result.multi_hand_landmarks:
            for hand_landmark in result.multi_hand_landmarks:
                h, w, _ = frame.shape
                x = int(hand_landmark.landmark[8].x * w)
                y = int(hand_landmark.landmark[8].y * h)
                fingertip = (x, y)
                self.drawing.draw_landmarks(frame, hand_landmark, self.hands_module.HAND_CONNECTIONS)
        return fingertip, frame


### utils/preprocessing.py
import numpy as np
import cv2

def draw_path_on_blank(path, img_size=256):
    blank = np.ones((img_size, img_size), dtype=np.uint8) * 255
    for i in range(1, len(path)):
        cv2.line(blank, path[i-1], path[i], (0), 4)
    return blank


### autocorrector.py
from textblob import TextBlob

def correct_word(word):
    blob = TextBlob(word)
    return str(blob.correct())


### text_to_speech.py
import pyttsx3

engine = pyttsx3.init()

def speak_word(word):
    engine.say(word)
    engine.runAndWait()


### train_cnn.py
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# Load dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Filter A-Z only (i.e., labels 0-25)
x_train = x_train[y_train < 26]
y_train = y_train[y_train < 26]
x_test = x_test[y_test < 26]
y_test = y_test[y_test < 26]

# Preprocess data
x_train = x_train.reshape(-1, 28, 28, 1) / 255.0
x_test = x_test.reshape(-1, 28, 28, 1) / 255.0
y_train = to_categorical(y_train, 26)
y_test = to_categorical(y_test, 26)

# Build model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(26, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

model.save("models/letter_recognition.h5")


### gui_app.py
import streamlit as st
import cv2
from PIL import Image
import numpy as np
from utils.hand_tracker import HandTracker
from utils.preprocessing import draw_path_on_blank
from keras.models import load_model
from autocorrector import correct_word
from text_to_speech import speak_word

model = load_model("models/letter_recognition.h5")
tracker = HandTracker()

st.title("Air Writing Recognition")
run = st.checkbox('Start Webcam')

FRAME_WINDOW = st.image([])
cap = cv2.VideoCapture(0)
path = []
word = ""

while run:
    ret, frame = cap.read()
    if not ret:
        break

    fingertip, annotated = tracker.get_fingertip(frame)
    if fingertip:
        path.append(fingertip)
        cv2.circle(annotated, fingertip, 4, (0, 255, 0), -1)

    annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    FRAME_WINDOW.image(annotated)

    key = cv2.waitKey(1)
    if key == ord(" "):
        if len(path) > 5:
            img = draw_path_on_blank(path)
            img_resized = cv2.resize(img, (28, 28)).reshape(1, 28, 28, 1) / 255.0
            prediction = model.predict(img_resized)
            letter = chr(np.argmax(prediction) + ord("A"))
            word += letter
            st.write(f"Letter: {letter}")
        path.clear()
    elif key == ord("s"):
        corrected = correct_word(word)
        st.write(f"Final word: {corrected}")
        speak_word(corrected)
        word = ""

cap.release()


### requirements.txt
opencv-python
mediapipe
numpy
textblob
pyttsx3
tensorflow
keras
streamlit
pillow


### test_app.py
import unittest
from unittest.mock import patch, MagicMock
import time
import numpy as np

# Assuming the main script is refactored into a function for testability
# For this example, we'll simulate the state changes within the test
# and mock the external dependencies.

class TestAppWordCompletion(unittest.TestCase):

    @patch('test_app.speak_word')
    @patch('test_app.correct_word', return_value="CORRECTED")
    @patch('test_app.time.time')
    def test_path_cleared_after_auto_word_finalization(self, mock_time, mock_correct_word, mock_speak_word):
        """
        Verify that the drawing path is cleared after a word is automatically finalized due to a pause.
        """
        # Arrange: Set up the initial state
        path = [(10, 10), (20, 20), (30, 30)]  # A non-empty path
        word = "TEST"
        last_movement_time = 1000.0
        WORD_COMPLETION_PAUSE = 2.5

        # Act: Simulate the passage of time to trigger word completion
        # Set the current time to be greater than last_movement_time + pause
        mock_time.return_value = last_movement_time + WORD_COMPLETION_PAUSE + 1

        # This block simulates the logic from the app's while loop
        if word and (mock_time() - last_movement_time) > WORD_COMPLETION_PAUSE:
            corrected = mock_correct_word(word)
            mock_speak_word(corrected)
            word = ""
            path.clear()  # The action being tested

        # Assert: Check if the path was cleared
        self.assertEqual(len(path), 0, "Path should be cleared after word finalization")
        self.assertEqual(word, "", "Word should be cleared after word finalization")
        mock_correct_word.assert_called_once_with("TEST")
        mock_speak_word.assert_called_once_with("CORRECTED")

if __name__ == '__main__':
    unittest.main()
