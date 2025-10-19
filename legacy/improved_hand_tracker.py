#!/usr/bin/env python3
"""
Improved Hand Tracker with Better Accuracy
Enhanced MediaPipe hand tracking with optimized settings for air writing
"""

import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import time
import math

class ImprovedHandTracker:
    def __init__(self):
        # Initialize MediaPipe with optimized settings
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=1,
            min_detection_confidence=0.7,  # Lowered for better detection
            min_tracking_confidence=0.5    # Lowered for smoother tracking
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Tracking variables
        self.fingertip_trail = deque(maxlen=200)  # Longer trail
        self.velocity_history = deque(maxlen=5)   # Shorter for responsiveness
        self.position_history = deque(maxlen=10)  # For smoothing
        
        # State variables
        self.current_position = None
        self.previous_position = None
        self.velocity = 0.0
        self.is_writing = False
        self.hand_detected = False
        
        # Smoothing parameters
        self.smoothing_factor = 0.7  # Higher = more smoothing
        self.velocity_threshold = 8.0  # Adjusted threshold
        
        # Writing detection
        self.writing_confidence = 0.0
        self.stable_frames = 0
        
        print("✅ Improved Hand Tracker initialized")
    
    def smooth_position(self, new_position):
        """Apply exponential smoothing to position"""
        if self.current_position is None:
            return new_position
        
        # Exponential moving average
        smoothed_x = (self.smoothing_factor * self.current_position[0] + 
                     (1 - self.smoothing_factor) * new_position[0])
        smoothed_y = (self.smoothing_factor * self.current_position[1] + 
                     (1 - self.smoothing_factor) * new_position[1])
        
        return (int(smoothed_x), int(smoothed_y))
    
    def calculate_velocity(self, current_pos, previous_pos):
        """Calculate velocity between two positions"""
        if previous_pos is None:
            return 0.0
        
        dx = current_pos[0] - previous_pos[0]
        dy = current_pos[1] - previous_pos[1]
        return math.sqrt(dx*dx + dy*dy)
    
    def detect_writing_gesture(self, hand_landmarks):
        """Improved writing gesture detection"""
        # Get landmark positions
        landmarks = hand_landmarks.landmark
        
        # Index finger landmarks
        index_tip = landmarks[8]
        index_pip = landmarks[6]
        index_mcp = landmarks[5]
        
        # Middle finger landmarks
        middle_tip = landmarks[12]
        middle_pip = landmarks[10]
        
        # Ring finger landmarks
        ring_tip = landmarks[16]
        ring_pip = landmarks[14]
        
        # Pinky landmarks
        pinky_tip = landmarks[20]
        pinky_pip = landmarks[18]
        
        # Thumb landmarks
        thumb_tip = landmarks[4]
        thumb_mcp = landmarks[2]
        
        # Check if index finger is extended
        index_extended = index_tip.y < index_pip.y < index_mcp.y
        
        # Check if other fingers are curled
        middle_curled = middle_tip.y > middle_pip.y
        ring_curled = ring_tip.y > ring_pip.y
        pinky_curled = pinky_tip.y > pinky_pip.y
        
        # Check thumb position (should be away from index)
        thumb_distance = math.sqrt(
            (thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2
        )
        thumb_ok = thumb_distance > 0.05
        
        # Calculate confidence
        confidence = 0.0
        if index_extended:
            confidence += 0.4
        if middle_curled:
            confidence += 0.2
        if ring_curled:
            confidence += 0.2
        if pinky_curled:
            confidence += 0.1
        if thumb_ok:
            confidence += 0.1
        
        return confidence > 0.7, confidence
    
    def update_trail(self, position):
        """Update the drawing trail with improved logic"""
        if position is None:
            return
        
        # Add to trail only if we're in writing mode
        if self.is_writing:
            self.fingertip_trail.append(position)
        
        # Update position history for smoothing
        self.position_history.append(position)
    
    def draw_enhanced_trail(self, frame):
        """Draw enhanced trail with better visualization"""
        if len(self.fingertip_trail) < 2:
            return
        
        trail_points = list(self.fingertip_trail)
        
        # Draw trail with gradient and variable thickness
        for i in range(1, len(trail_points)):
            # Calculate alpha for gradient
            alpha = i / len(trail_points)
            
            # Color gradient: Blue -> Green -> Yellow -> Red
            if alpha < 0.33:
                # Blue to Green
                color = (int(255 * (1 - alpha * 3)), int(255 * alpha * 3), 0)
            elif alpha < 0.66:
                # Green to Yellow
                local_alpha = (alpha - 0.33) * 3
                color = (0, 255, int(255 * local_alpha))
            else:
                # Yellow to Red
                local_alpha = (alpha - 0.66) * 3
                color = (0, int(255 * (1 - local_alpha)), 255)
            
            # Variable thickness based on position in trail
            thickness = max(2, int(6 * alpha))
            
            cv2.line(frame, trail_points[i-1], trail_points[i], color, thickness)
        
        # Draw direction indicators
        if len(trail_points) >= 6:
            recent_points = trail_points[-6:]
            for i in range(1, len(recent_points), 2):
                start = recent_points[i-1]
                end = recent_points[i]
                
                # Calculate direction vector
                dx = end[0] - start[0]
                dy = end[1] - start[1]
                length = math.sqrt(dx*dx + dy*dy)
                
                if length > 5:  # Only draw if movement is significant
                    # Normalize and extend
                    dx, dy = dx/length, dy/length
                    arrow_end = (int(end[0] + dx * 12), int(end[1] + dy * 12))
                    
                    cv2.arrowedLine(frame, end, arrow_end, (0, 255, 255), 2, tipLength=0.4)
    
    def process_frame(self, frame):
        """Process a single frame and return tracking results"""
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = self.hands.process(rgb_frame)
        
        # Reset detection flag
        self.hand_detected = False
        fingertip_position = None
        
        if results.multi_hand_landmarks:
            self.hand_detected = True
            
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
                
                # Get index fingertip position
                h, w, _ = frame.shape
                index_tip = hand_landmarks.landmark[8]
                raw_position = (int(index_tip.x * w), int(index_tip.y * h))
                
                # Detect writing gesture
                is_writing_gesture, confidence = self.detect_writing_gesture(hand_landmarks)
                self.writing_confidence = confidence
                
                if is_writing_gesture:
                    # Apply smoothing
                    fingertip_position = self.smooth_position(raw_position)
                    
                    # Calculate velocity
                    if self.previous_position is not None:
                        current_velocity = self.calculate_velocity(fingertip_position, self.previous_position)
                        self.velocity_history.append(current_velocity)
                        
                        # Average velocity over recent frames
                        if len(self.velocity_history) > 0:
                            self.velocity = sum(self.velocity_history) / len(self.velocity_history)
                    
                    # Determine if we're actively writing
                    if self.velocity > self.velocity_threshold:
                        self.is_writing = True
                        self.stable_frames = 0
                    else:
                        self.stable_frames += 1
                        if self.stable_frames > 10:  # Stop writing after 10 stable frames
                            self.is_writing = False
                    
                    # Update positions
                    self.previous_position = self.current_position
                    self.current_position = fingertip_position
                    
                    # Update trail
                    self.update_trail(fingertip_position)
                    
                    # Draw fingertip indicator
                    color = (0, 0, 255) if self.is_writing else (0, 255, 0)  # Red if writing, green if not
                    cv2.circle(frame, fingertip_position, 12, color, -1)
                    cv2.circle(frame, fingertip_position, 15, (255, 255, 255), 2)
                    
                    # Draw writing status
                    status_text = "WRITING" if self.is_writing else "READY"
                    cv2.putText(frame, status_text, (fingertip_position[0] + 20, fingertip_position[1] - 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        else:
            # No hand detected - reset everything
            self.reset_tracking()
        
        # Draw enhanced trail
        self.draw_enhanced_trail(frame)
        
        # Draw status information
        self.draw_status_info(frame)
        
        return fingertip_position, self.velocity, self.is_writing
    
    def draw_status_info(self, frame):
        """Draw status information on frame"""
        h, w = frame.shape[:2]
        
        # Status panel
        panel_height = 120
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Status text
        status_info = [
            f"Hand Detected: {'Yes' if self.hand_detected else 'No'}",
            f"Writing Mode: {'Yes' if self.is_writing else 'No'}",
            f"Velocity: {self.velocity:.1f}",
            f"Writing Confidence: {self.writing_confidence:.2f}",
            f"Trail Points: {len(self.fingertip_trail)}"
        ]
        
        for i, info in enumerate(status_info):
            color = (0, 255, 0) if self.hand_detected else (0, 0, 255)
            cv2.putText(frame, info, (20, 30 + i * 18), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    def reset_tracking(self):
        """Reset all tracking variables"""
        self.current_position = None
        self.previous_position = None
        self.velocity = 0.0
        self.is_writing = False
        self.writing_confidence = 0.0
        self.stable_frames = 0
        self.velocity_history.clear()
        self.position_history.clear()
    
    def get_current_path(self):
        """Get the current drawing path"""
        return list(self.fingertip_trail)
    
    def clear_trail(self):
        """Clear the current trail"""
        self.fingertip_trail.clear()
    
    def is_hand_detected(self):
        """Check if hand is currently detected"""
        return self.hand_detected
    
    def is_actively_writing(self):
        """Check if actively writing"""
        return self.is_writing and self.hand_detected

def test_improved_tracker():
    """Test the improved hand tracker"""
    print("🧪 Testing Improved Hand Tracker...")
    
    tracker = ImprovedHandTracker()
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Could not open camera")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    print("📋 Instructions:")
    print("   - Hold up your index finger (other fingers curled)")
    print("   - Move your finger to write in the air")
    print("   - Press ESC to exit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Flip frame horizontally
        frame = cv2.flip(frame, 1)
        
        # Process frame
        fingertip, velocity, is_writing = tracker.process_frame(frame)
        
        # Show frame
        cv2.imshow("Improved Hand Tracker Test", frame)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord('c'):  # Clear trail
            tracker.clear_trail()
    
    cap.release()
    cv2.destroyAllWindows()
    print("✅ Test completed")

if __name__ == "__main__":
    test_improved_tracker()