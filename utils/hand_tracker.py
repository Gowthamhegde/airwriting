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
