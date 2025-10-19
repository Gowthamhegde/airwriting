#!/usr/bin/env python3
"""
MediaPipe Hand Tracking System
Pure hand movement tracking using MediaPipe
"""

import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque
import math

class MediaPipeHandTracker:
    """MediaPipe-based hand tracker for movement detection"""
    
    def __init__(self):
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Hand detection settings
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # Movement tracking
        self.trail_points = deque(maxlen=50)
        self.fingertip_trail = deque(maxlen=30)
        self.velocity_history = deque(maxlen=10)
        
        # Hand landmarks
        self.current_landmarks = None
        self.fingertip_position = None
        self.palm_center = None
        
        # Movement analysis
        self.is_writing = False
        self.movement_threshold = 10
        self.writing_velocity_threshold = 15
        
        # Gesture detection
        self.finger_states = {'thumb': False, 'index': False, 'middle': False, 'ring': False, 'pinky': False}
        self.gesture_name = "Unknown"
        
    def detect_finger_states(self, landmarks):
        """Detect which fingers are extended"""
        if not landmarks:
            return
        
        # Finger tip and pip landmarks
        finger_tips = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky
        finger_pips = [3, 6, 10, 14, 18]
        
        fingers = []
        
        # Thumb (different logic due to orientation)
        if landmarks[finger_tips[0]].x > landmarks[finger_pips[0]].x:
            fingers.append(1)
        else:
            fingers.append(0)
        
        # Other fingers
        for i in range(1, 5):
            if landmarks[finger_tips[i]].y < landmarks[finger_pips[i]].y:
                fingers.append(1)
            else:
                fingers.append(0)
        
        # Update finger states
        finger_names = ['thumb', 'index', 'middle', 'ring', 'pinky']
        for i, name in enumerate(finger_names):
            self.finger_states[name] = bool(fingers[i])
        
        # Determine gesture
        self.determine_gesture(fingers)
        
        return fingers
    
    def determine_gesture(self, fingers):
        """Determine hand gesture based on finger states"""
        total_fingers = sum(fingers)
        
        if total_fingers == 0:
            self.gesture_name = "Fist"
        elif total_fingers == 1:
            if fingers[1]:  # Index finger only
                self.gesture_name = "Pointing"
            elif fingers[0]:  # Thumb only
                self.gesture_name = "Thumbs Up"
            else:
                self.gesture_name = "One Finger"
        elif total_fingers == 2:
            if fingers[1] and fingers[2]:  # Index and middle
                self.gesture_name = "Peace/Victory"
            elif fingers[0] and fingers[1]:  # Thumb and index
                self.gesture_name = "Gun/L-Shape"
            else:
                self.gesture_name = "Two Fingers"
        elif total_fingers == 5:
            self.gesture_name = "Open Hand"
        else:
            self.gesture_name = f"{total_fingers} Fingers"
    
    def get_fingertip_position(self, landmarks, frame_shape):
        """Get index fingertip position in pixel coordinates"""
        if landmarks:
            # Index finger tip (landmark 8)
            index_tip = landmarks[8]
            h, w = frame_shape[:2]
            x = int(index_tip.x * w)
            y = int(index_tip.y * h)
            return (x, y)
        return None
    
    def get_palm_center(self, landmarks, frame_shape):
        """Get palm center position"""
        if landmarks:
            # Use wrist (0) and middle finger MCP (9) to estimate palm center
            wrist = landmarks[0]
            middle_mcp = landmarks[9]
            
            h, w = frame_shape[:2]
            center_x = int((wrist.x + middle_mcp.x) / 2 * w)
            center_y = int((wrist.y + middle_mcp.y) / 2 * h)
            return (center_x, center_y)
        return None
    
    def calculate_hand_velocity(self):
        """Calculate hand movement velocity"""
        if len(self.fingertip_trail) < 2:
            return 0
        
        # Calculate distance between last two positions
        p1 = self.fingertip_trail[-2]
        p2 = self.fingertip_trail[-1]
        
        distance = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
        self.velocity_history.append(distance)
        
        # Calculate average velocity
        avg_velocity = np.mean(self.velocity_history) if self.velocity_history else 0
        
        # Determine if writing motion
        self.is_writing = avg_velocity > self.writing_velocity_threshold
        
        return avg_velocity
    
    def process_frame(self, frame):
        """Process frame and detect hand landmarks"""
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = self.hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            # Get first hand
            hand_landmarks = results.multi_hand_landmarks[0]
            self.current_landmarks = hand_landmarks.landmark
            
            # Get fingertip and palm positions
            self.fingertip_position = self.get_fingertip_position(self.current_landmarks, frame.shape)
            self.palm_center = self.get_palm_center(self.current_landmarks, frame.shape)
            
            # Detect finger states and gestures
            self.detect_finger_states(self.current_landmarks)
            
            # Update trails
            if self.fingertip_position:
                # Add to fingertip trail
                if (not self.fingertip_trail or 
                    math.sqrt((self.fingertip_position[0] - self.fingertip_trail[-1][0])**2 + 
                             (self.fingertip_position[1] - self.fingertip_trail[-1][1])**2) > self.movement_threshold):
                    self.fingertip_trail.append(self.fingertip_position)
                
                # Add to main trail if index finger is extended (pointing gesture)
                if self.finger_states['index'] and not self.finger_states['middle']:
                    self.trail_points.append(self.fingertip_position)
            
            # Calculate velocity
            velocity = self.calculate_hand_velocity()
            
            return hand_landmarks, velocity
        else:
            self.current_landmarks = None
            self.fingertip_position = None
            self.palm_center = None
            return None, 0

class HandTrackingSystem:
    """Main MediaPipe hand tracking system"""
    
    def __init__(self):
        self.tracker = MediaPipeHandTracker()
        self.cap = None
        self.running = False
        
        # Display settings
        self.show_landmarks = True
        self.show_trail = True
        self.show_fingertip_trail = True
        self.show_connections = True
        self.show_info = True
        
        # Colors
        self.trail_color = (0, 255, 0)  # Green
        self.fingertip_trail_color = (255, 0, 0)  # Red
        self.fingertip_color = (0, 0, 255)  # Red
        self.palm_color = (255, 255, 0)  # Cyan
        
    def initialize_camera(self):
        """Initialize camera"""
        print("🎥 Initializing camera...")
        self.cap = cv2.VideoCapture(0)
        
        if not self.cap.isOpened():
            print("❌ Could not open camera")
            return False
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("✅ Camera initialized")
        return True
    
    def draw_trails(self, frame):
        """Draw movement trails"""
        # Draw main trail (writing trail)
        if self.show_trail and len(self.tracker.trail_points) > 1:
            for i in range(1, len(self.tracker.trail_points)):
                thickness = max(1, int((i / len(self.tracker.trail_points)) * 5))
                cv2.line(frame, 
                        self.tracker.trail_points[i-1], 
                        self.tracker.trail_points[i], 
                        self.trail_color, 
                        thickness)
        
        # Draw fingertip trail
        if self.show_fingertip_trail and len(self.tracker.fingertip_trail) > 1:
            for i in range(1, len(self.tracker.fingertip_trail)):
                alpha = i / len(self.tracker.fingertip_trail)
                thickness = max(1, int(alpha * 3))
                cv2.line(frame, 
                        self.tracker.fingertip_trail[i-1], 
                        self.tracker.fingertip_trail[i], 
                        self.fingertip_trail_color, 
                        thickness)
    
    def draw_hand_info(self, frame, velocity):
        """Draw hand information overlay"""
        if not self.show_info:
            return
        
        height, width = frame.shape[:2]
        
        # Hand detection status
        if self.tracker.current_landmarks:
            status_text = "✅ Hand Detected"
            status_color = (0, 255, 0)
        else:
            status_text = "❌ No Hand"
            status_color = (0, 0, 255)
        
        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        if self.tracker.current_landmarks:
            # Gesture information
            gesture_text = f"Gesture: {self.tracker.gesture_name}"
            cv2.putText(frame, gesture_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Finger states
            finger_text = "Fingers: "
            for name, state in self.tracker.finger_states.items():
                finger_text += f"{name[0].upper()}{'✓' if state else '✗'} "
            cv2.putText(frame, finger_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Position information
            if self.tracker.fingertip_position:
                pos_text = f"Fingertip: {self.tracker.fingertip_position}"
                cv2.putText(frame, pos_text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Velocity and writing status
            vel_text = f"Velocity: {velocity:.1f}"
            cv2.putText(frame, vel_text, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            writing_text = "✍️ Writing Motion" if self.tracker.is_writing else "✋ Stationary"
            writing_color = (0, 255, 255) if self.tracker.is_writing else (255, 255, 255)
            cv2.putText(frame, writing_text, (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, writing_color, 1)
        
        # Controls
        controls = [
            "Controls:",
            "L - Toggle landmarks",
            "T - Toggle trail", 
            "F - Toggle fingertip trail",
            "C - Toggle connections",
            "I - Toggle info",
            "R - Reset trails",
            "ESC - Exit"
        ]
        
        for i, control in enumerate(controls):
            cv2.putText(frame, control, (width - 250, 30 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    def draw_landmarks_and_connections(self, frame, hand_landmarks):
        """Draw hand landmarks and connections"""
        if not hand_landmarks:
            return
        
        # Draw landmarks
        if self.show_landmarks:
            self.tracker.mp_drawing.draw_landmarks(
                frame, 
                hand_landmarks,
                self.tracker.mp_hands.HAND_CONNECTIONS if self.show_connections else None,
                self.tracker.mp_drawing_styles.get_default_hand_landmarks_style(),
                self.tracker.mp_drawing_styles.get_default_hand_connections_style() if self.show_connections else None
            )
        
        # Draw special points
        if self.tracker.fingertip_position:
            cv2.circle(frame, self.tracker.fingertip_position, 8, self.fingertip_color, -1)
            cv2.circle(frame, self.tracker.fingertip_position, 12, (255, 255, 255), 2)
        
        if self.tracker.palm_center:
            cv2.circle(frame, self.tracker.palm_center, 6, self.palm_color, -1)
    
    def run(self):
        """Run the MediaPipe hand tracking system"""
        if not self.initialize_camera():
            return
        
        print("\n🖐️  MEDIAPIPE HAND TRACKING SYSTEM")
        print("=" * 60)
        print("📋 Instructions:")
        print("   • Hold your hand in front of the camera")
        print("   • Point with index finger to draw trail")
        print("   • Try different gestures (fist, peace, open hand)")
        print("   • Red trail shows fingertip movement")
        print("   • Green trail shows writing motion")
        print("\n⌨️  Controls:")
        print("   L - Toggle landmarks display")
        print("   T - Toggle writing trail")
        print("   F - Toggle fingertip trail")
        print("   C - Toggle hand connections")
        print("   I - Toggle info overlay")
        print("   R - Reset all trails")
        print("   ESC - Exit")
        print("=" * 60)
        
        self.running = True
        fps_counter = 0
        fps_start_time = time.time()
        
        try:
            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    print("❌ Failed to read frame")
                    break
                
                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Process frame with MediaPipe
                hand_landmarks, velocity = self.tracker.process_frame(frame)
                
                # Draw trails
                self.draw_trails(frame)
                
                # Draw hand landmarks and connections
                self.draw_landmarks_and_connections(frame, hand_landmarks)
                
                # Draw information overlay
                self.draw_hand_info(frame, velocity)
                
                # Show frame
                cv2.imshow('MediaPipe Hand Tracking', frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == 27:  # ESC
                    break
                elif key == ord('l') or key == ord('L'):
                    self.show_landmarks = not self.show_landmarks
                    print(f"Landmarks display: {'ON' if self.show_landmarks else 'OFF'}")
                elif key == ord('t') or key == ord('T'):
                    self.show_trail = not self.show_trail
                    print(f"Writing trail: {'ON' if self.show_trail else 'OFF'}")
                elif key == ord('f') or key == ord('F'):
                    self.show_fingertip_trail = not self.show_fingertip_trail
                    print(f"Fingertip trail: {'ON' if self.show_fingertip_trail else 'OFF'}")
                elif key == ord('c') or key == ord('C'):
                    self.show_connections = not self.show_connections
                    print(f"Hand connections: {'ON' if self.show_connections else 'OFF'}")
                elif key == ord('i') or key == ord('I'):
                    self.show_info = not self.show_info
                    print(f"Info overlay: {'ON' if self.show_info else 'OFF'}")
                elif key == ord('r') or key == ord('R'):
                    self.tracker.trail_points.clear()
                    self.tracker.fingertip_trail.clear()
                    print("All trails reset")
                
                # Calculate and display FPS
                fps_counter += 1
                if fps_counter % 30 == 0:
                    fps = 30 / (time.time() - fps_start_time)
                    print(f"📊 FPS: {fps:.1f} | Gesture: {self.tracker.gesture_name}")
                    fps_start_time = time.time()
        
        except KeyboardInterrupt:
            print("\n👋 System interrupted by user")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        self.running = False
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("🧹 Cleanup completed")

def main():
    """Main function"""
    print("🚀 Starting MediaPipe Hand Tracking System...")
    
    try:
        system = HandTrackingSystem()
        system.run()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("👋 MediaPipe Hand Tracking System stopped")

if __name__ == "__main__":
    main()