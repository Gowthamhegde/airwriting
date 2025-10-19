import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import time

class HandTracker:
    def __init__(self, max_hands=1, trail_length=100, alpha=0.3):
        self.hands_module = mp.solutions.hands
        self.hands = self.hands_module.Hands(
            max_num_hands=max_hands, 
            min_detection_confidence=0.8,
            min_tracking_confidence=0.7,
            model_complexity=1
        )
        self.drawing = mp.solutions.drawing_utils
        self.drawing_styles = mp.solutions.drawing_styles
        
        # Tracking variables
        self.fingertip_trail = deque(maxlen=trail_length)
        self.velocity_history = deque(maxlen=10)
        self.prev_smoothed_x = None
        self.prev_smoothed_y = None
        self.velocity = 0
        self.alpha = alpha  # EMA smoothing factor
        
        # Gesture recognition
        self.gesture_history = deque(maxlen=30)
        self.last_gesture_time = time.time()
        
        # Hand state
        self.hand_present = False
        self.writing_mode = False
        self.gesture_confidence = 0.0

    def ema_smooth(self, current, previous):
        """Exponential Moving Average smoothing"""
        return self.alpha * current + (1 - self.alpha) * previous

    def calculate_finger_distances(self, hand_landmark):
        """Calculate distances between fingertips and palm"""
        # Palm center (approximate)
        palm_x = hand_landmark.landmark[0].x
        palm_y = hand_landmark.landmark[0].y
        
        # Fingertip landmarks
        fingertips = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky
        distances = []
        
        for tip_idx in fingertips:
            tip_x = hand_landmark.landmark[tip_idx].x
            tip_y = hand_landmark.landmark[tip_idx].y
            distance = np.sqrt((tip_x - palm_x)**2 + (tip_y - palm_y)**2)
            distances.append(distance)
        
        return distances

    def detect_writing_gesture(self, hand_landmark):
        """Detect if hand is in writing position (index finger extended)"""
        # Get finger tip and joint positions
        index_tip = hand_landmark.landmark[8]
        index_pip = hand_landmark.landmark[6]
        index_mcp = hand_landmark.landmark[5]
        
        middle_tip = hand_landmark.landmark[12]
        middle_pip = hand_landmark.landmark[10]
        
        ring_tip = hand_landmark.landmark[16]
        ring_pip = hand_landmark.landmark[14]
        
        pinky_tip = hand_landmark.landmark[20]
        pinky_pip = hand_landmark.landmark[18]
        
        # Check if index finger is extended (tip above joints)
        index_extended = (index_tip.y < index_pip.y < index_mcp.y)
        
        # Check if other fingers are curled (tips below joints)
        middle_curled = middle_tip.y > middle_pip.y
        ring_curled = ring_tip.y > ring_pip.y
        pinky_curled = pinky_tip.y > pinky_pip.y
        
        # Calculate confidence based on finger positions
        confidence = 0.0
        if index_extended:
            confidence += 0.4
        if middle_curled:
            confidence += 0.2
        if ring_curled:
            confidence += 0.2
        if pinky_curled:
            confidence += 0.2
        
        return confidence > 0.6, confidence

    def detect_gesture_commands(self, hand_landmark):
        """Detect special gesture commands"""
        # Get all fingertip positions
        fingertips = []
        for i in [4, 8, 12, 16, 20]:  # Thumb, Index, Middle, Ring, Pinky
            fingertips.append(hand_landmark.landmark[i])
        
        # Detect "OK" gesture (thumb and index touching, others extended)
        thumb_index_distance = np.sqrt(
            (fingertips[0].x - fingertips[1].x)**2 + 
            (fingertips[0].y - fingertips[1].y)**2
        )
        
        if thumb_index_distance < 0.05:  # Threshold for "touching"
            return "OK"
        
        # Detect "Peace" gesture (index and middle extended, others curled)
        index_extended = fingertips[1].y < hand_landmark.landmark[6].y
        middle_extended = fingertips[2].y < hand_landmark.landmark[10].y
        ring_curled = fingertips[3].y > hand_landmark.landmark[14].y
        pinky_curled = fingertips[4].y > hand_landmark.landmark[18].y
        
        if index_extended and middle_extended and ring_curled and pinky_curled:
            return "PEACE"
        
        # Detect closed fist (all fingers curled)
        all_curled = all(
            fingertips[i].y > hand_landmark.landmark[j].y 
            for i, j in [(1, 6), (2, 10), (3, 14), (4, 18)]
        )
        
        if all_curled:
            return "FIST"
        
        return "NONE"

    def update_velocity(self):
        """Update velocity with smoothing"""
        if len(self.velocity_history) > 0:
            self.velocity = np.mean(list(self.velocity_history))
        else:
            self.velocity = 0

    def get_fingertip(self, frame):
        """Enhanced fingertip detection with gesture recognition"""
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.hands.process(img_rgb)
        
        fingertip = None
        smoothed_fingertip = None
        gesture = "NONE"
        
        if result.multi_hand_landmarks:
            self.hand_present = True
            
            for hand_idx, hand_landmark in enumerate(result.multi_hand_landmarks):
                h, w, _ = frame.shape
                
                # Get handedness
                handedness = "Right"
                if result.multi_handedness:
                    handedness = result.multi_handedness[hand_idx].classification[0].label
                
                # Detect writing gesture
                is_writing, confidence = self.detect_writing_gesture(hand_landmark)
                self.gesture_confidence = confidence
                
                # Detect other gestures
                gesture = self.detect_gesture_commands(hand_landmark)
                
                # Get index fingertip position
                x = int(hand_landmark.landmark[8].x * w)
                y = int(hand_landmark.landmark[8].y * h)
                fingertip = (x, y)
                
                # Only track fingertip if in writing mode
                if is_writing:
                    self.writing_mode = True
                    
                    # Draw enhanced hand landmarks
                    self.drawing.draw_landmarks(
                        frame, hand_landmark, self.hands_module.HAND_CONNECTIONS,
                        self.drawing_styles.get_default_hand_landmarks_style(),
                        self.drawing_styles.get_default_hand_connections_style()
                    )
                    
                    # Apply EMA smoothing
                    if self.prev_smoothed_x is None:
                        smoothed_x, smoothed_y = x, y
                    else:
                        smoothed_x = self.ema_smooth(x, self.prev_smoothed_x)
                        smoothed_y = self.ema_smooth(y, self.prev_smoothed_y)
                    
                    smoothed_fingertip = (int(smoothed_x), int(smoothed_y))
                    
                    # Calculate velocity
                    if self.prev_smoothed_x is not None:
                        dx = smoothed_x - self.prev_smoothed_x
                        dy = smoothed_y - self.prev_smoothed_y
                        current_velocity = np.sqrt(dx*dx + dy*dy)
                        self.velocity_history.append(current_velocity)
                        self.update_velocity()
                    
                    # Update previous position
                    self.prev_smoothed_x, self.prev_smoothed_y = smoothed_x, smoothed_y
                    
                    # Add to trail
                    self.fingertip_trail.append(smoothed_fingertip)
                    
                    # Draw enhanced trail
                    self.draw_enhanced_trail(frame)
                    
                    # Draw fingertip with status indicator
                    is_moving = self.velocity > 3.0
                    color = (0, 0, 255) if is_moving else (0, 255, 0)  # Red if moving, green if stationary
                    cv2.circle(frame, smoothed_fingertip, 10, color, -1)
                    cv2.circle(frame, smoothed_fingertip, 12, (255, 255, 255), 2)
                    
                else:
                    # Not in writing mode
                    self.writing_mode = False
                    fingertip = None
                    
                    # Draw basic hand landmarks
                    self.drawing.draw_landmarks(
                        frame, hand_landmark, self.hands_module.HAND_CONNECTIONS
                    )
                
                # Display hand info
                info_y = 30 + hand_idx * 120
                cv2.putText(frame, f"Hand: {handedness}", (10, info_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(frame, f"Writing: {'Yes' if is_writing else 'No'}", 
                           (10, info_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                           (0, 255, 0) if is_writing else (0, 0, 255), 2)
                cv2.putText(frame, f"Confidence: {confidence:.2f}", 
                           (10, info_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                           (255, 255, 255), 2)
                cv2.putText(frame, f"Gesture: {gesture}", 
                           (10, info_y + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                           (255, 255, 0), 2)
        
        else:
            # No hand detected - reset everything
            self.hand_present = False
            self.writing_mode = False
            self.fingertip_trail.clear()
            self.velocity_history.clear()
            self.prev_smoothed_x = None
            self.prev_smoothed_y = None
            self.velocity = 0
            self.gesture_confidence = 0.0
        
        return smoothed_fingertip, frame, self.velocity

    def draw_enhanced_trail(self, frame):
        """Draw enhanced trail with gradient and thickness variation"""
        if len(self.fingertip_trail) < 2:
            return
        
        trail_points = list(self.fingertip_trail)
        
        for i in range(1, len(trail_points)):
            # Calculate alpha for gradient effect
            alpha = i / len(trail_points)
            
            # Calculate thickness based on velocity at that point
            thickness = max(1, min(8, int(4 + self.velocity / 5)))
            
            # Color gradient from blue to red
            blue_component = int(255 * (1 - alpha))
            red_component = int(255 * alpha)
            trail_color = (blue_component, 0, red_component)
            
            cv2.line(frame, trail_points[i-1], trail_points[i], trail_color, thickness)
        
        # Draw direction arrows for recent movement
        if len(trail_points) >= 10:
            recent_points = trail_points[-10:]
            for i in range(2, len(recent_points), 3):
                start_point = recent_points[i-2]
                end_point = recent_points[i]
                
                # Calculate arrow direction
                dx = end_point[0] - start_point[0]
                dy = end_point[1] - start_point[1]
                
                if dx != 0 or dy != 0:
                    # Normalize direction
                    length = np.sqrt(dx*dx + dy*dy)
                    if length > 0:
                        dx, dy = dx/length, dy/length
                        
                        # Draw small arrow
                        arrow_length = 15
                        arrow_end = (int(end_point[0] + dx * arrow_length),
                                   int(end_point[1] + dy * arrow_length))
                        
                        cv2.arrowedLine(frame, end_point, arrow_end, 
                                      (0, 255, 255), 2, tipLength=0.3)

    def get_trail_path(self):
        """Get the current trail as a list of points"""
        return list(self.fingertip_trail)
    
    def clear_trail(self):
        """Clear the current trail"""
        self.fingertip_trail.clear()
    
    def is_writing_active(self):
        """Check if currently in writing mode"""
        return self.writing_mode and self.hand_present
