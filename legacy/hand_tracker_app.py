import cv2
import mediapipe as mp
import numpy as np
from collections import deque

# Function for Exponential Moving Average smoothing
def ema_smooth(current, previous, alpha=0.3):
    return alpha * current + (1 - alpha) * previous

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize webcam
cap = cv2.VideoCapture(0)

# Initialize variables for tracking
fingertip_trail = deque(maxlen=64)  # Store last 64 points
prev_smoothed_x, prev_smoothed_y = None, None
velocity = 0

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Flip the image horizontally for a selfie-view display
        image = cv2.flip(image, 1)
        
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        # Draw the hand annotations on the image
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Reset trail if no hands detected
        if not results.multi_hand_landmarks:
            fingertip_trail.clear()
            prev_smoothed_x, prev_smoothed_y = None, None
            velocity = 0
        
        if results.multi_hand_landmarks:
            # Process each detected hand
            for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Draw landmarks and connections
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
                
                # Get handedness (left or right hand)
                handedness = results.multi_handedness[hand_idx].classification[0].label
                
                # Get index fingertip coordinates (landmark 8)
                index_fingertip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                h, w, c = image.shape
                x, y = int(index_fingertip.x * w), int(index_fingertip.y * h)
                
                # Apply EMA smoothing
                if prev_smoothed_x is None:
                    smoothed_x, smoothed_y = x, y
                else:
                    smoothed_x = ema_smooth(x, prev_smoothed_x)
                    smoothed_y = ema_smooth(y, prev_smoothed_y)
                
                # Calculate velocity (pixels/frame)
                if prev_smoothed_x is not None:
                    dx = smoothed_x - prev_smoothed_x
                    dy = smoothed_y - prev_smoothed_y
                    velocity = np.sqrt(dx*dx + dy*dy)  # L2 distance
                
                # Update previous position
                prev_smoothed_x, prev_smoothed_y = smoothed_x, smoothed_y
                
                # Add to trail
                fingertip_trail.append((int(smoothed_x), int(smoothed_y)))
                
                # Draw colored dot at fingertip (red if moving, green if not)
                is_moving = velocity > 5.0
                color = (0, 0, 255) if is_moving else (0, 255, 0)  # BGR format
                cv2.circle(image, (int(smoothed_x), int(smoothed_y)), 8, color, -1)
                
                # Draw trail
                if len(fingertip_trail) > 1:
                    for i in range(1, len(fingertip_trail)):
                        # Gradually change color based on position in trail
                        alpha = i / len(fingertip_trail)
                        trail_color = (int(255 * (1-alpha)), int(255 * alpha), 0)  # BGR format
                        cv2.line(image, fingertip_trail[i-1], fingertip_trail[i], trail_color, 2)
                
                # Display handedness
                cv2.putText(image, f"Hand: {handedness}", (10, 30 + hand_idx * 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Display velocity and movement status
        moving_text = "Moving: Yes" if velocity > 5.0 else "Moving: No"
        cv2.putText(image, moving_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(image, f"Velocity: {velocity:.1f} px/frame", (10, 120), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Display the resulting frame
        cv2.imshow('MediaPipe Hands', image)
        
        # Check for 'q' key to exit
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

# Release resources
cap.release()
cv2.destroyAllWindows()