### main.py
import cv2
import time
import numpy as np
from keras.models import load_model
from utils.preprocessing import draw_path_on_blank
from autocorrector import correct_word
from text_to_speech import speak_word

model = load_model("models/letter_recognition.h5")

cap = cv2.VideoCapture(0)
path = []
word = ""
last_point_time = time.time()
prediction_delay = 1.5  # seconds

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    fingertip = None
    if contours:
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) > 1000:
            M = cv2.moments(largest)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                fingertip = (cx, cy)
                cv2.circle(frame, fingertip, 4, (0, 255, 0), -1)

    now = time.time()
    if fingertip:
        path.append(fingertip)
        last_point_time = now

    if len(path) > 5 and now - last_point_time > prediction_delay:
        img = draw_path_on_blank(path)
        img_resized = cv2.resize(img, (28, 28)).reshape(1, 28, 28, 1) / 255.0
        prediction = model.predict(img_resized)
        letter = chr(np.argmax(prediction) + ord("A"))
        word += letter
        print(f"Letter: {letter}")
        path.clear()

    cv2.putText(frame, f"Word: {word}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow("AirWrite", frame)

    key = cv2.waitKey(1)
    if key == ord("s"):
        corrected = correct_word(word)
        print(f"Final word: {corrected}")
        speak_word(corrected)
        word = ""
    elif key == ord("c"):
        word = ""
    elif key == 27:
        break

cap.release()
cv2.destroyAllWindows()


### gui_app.py
import streamlit as st
import cv2
import numpy as np
import time
from PIL import Image
from utils.preprocessing import draw_path_on_blank
from keras.models import load_model
from autocorrector import correct_word
from text_to_speech import speak_word

model = load_model("models/letter_recognition.h5")

st.title("Air Writing Recognition")
run = st.checkbox('Start Webcam')
FRAME_WINDOW = st.image([])

path = []
word = ""
cap = cv2.VideoCapture(0)
last_point_time = time.time()
prediction_delay = 1.5

while run:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    fingertip = None
    if contours:
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) > 1000:
            M = cv2.moments(largest)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                fingertip = (cx, cy)
                cv2.circle(frame, fingertip, 4, (0, 255, 0), -1)

    now = time.time()
    if fingertip:
        path.append(fingertip)
        last_point_time = now

    if len(path) > 5 and now - last_point_time > prediction_delay:
        img = draw_path_on_blank(path)
        img_resized = cv2.resize(img, (28, 28)).reshape(1, 28, 28, 1) / 255.0
        prediction = model.predict(img_resized)
        letter = chr(np.argmax(prediction) + ord("A"))
        word += letter
        st.write(f"Predicted Letter: {letter}")
        path.clear()

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    FRAME_WINDOW.image(frame_rgb)
    st.write(f"Current Word: {word}")

    if st.button("Speak Word"):
        corrected = correct_word(word)
        st.write(f"Corrected Word: {corrected}")
        speak_word(corrected)
        word = ""

    if st.button("Clear Word"):
        word = ""

cap.release()