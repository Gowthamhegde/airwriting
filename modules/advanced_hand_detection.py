#!/usr/bin/env python3
"""
Advanced Hand Detection Module
Features:
- Precise open/closed hand detection
- Writing gesture recognition
- Hand state tracking with stability
- Multiple gesture commands
- Real-time performance optimization
"""

import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque
from enum import Enum

class HandState(Enum):
    """Hand state enumeration"""
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    WRITING = "WRITING"
    POINTING = "POINTING"
    PEACE = "PEACE"
    OK = "OK"
    UNKNOWN = "UNKNOWN"

class AdvancedHandDetector:
    """Advanced hand detector with precise gesture recognition"""
    
    def __init__(self, max_hands=1, detection_confidence=0.8, tracking_confidence=0.7):
        # Initialize MediaPipe
        self.hands_module = mp.solutions.hands
        self.hands = self.hands_module.Hands(
            max_num_hands=max_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
            model_complexity=1
        )
        self.drawing = mp.solutions.drawing_utils
        self.drawing_styles = mp.solutions.drawing_styles
        
        # State tracking
        self.current_state = HandState.UNKNOWN
        self.state_history = deque(maxlen=10)
        self.confidence_history = deque(maxlen=10)
        self.last_state_change = time.time()
        self.state_stability_threshold = 0.3  # seconds
        
        # Hand landmarks and metrics
        self.hand_landmarks = None
        self.hand_present = False
        self.handedness = "Unknown"
        
        # Finger tracking
        self.finger_states = {
            'thumb': False,
            'index': False,
            'middle': False,
            'ring': False,
            'pinky': False
        }
        
        # Performance tracking
        self.processing_times = deque(maxlen=30)
        
        print("✅ Advanced hand detector initialized")
    
    def calculate_finger_extension(self, landmarks):
        """
        Calculate finger extension states with improved accuracy
        Returns: dict of finger states and confidence scores
        """
        finger_states = {}
        finger_confidences = {}
        
        # Landmark indices for fingertips and joints
        finger_landmarks = {
            'thumb': {'tip': 4, 'pip': 3, 'mcp': 2, 'cmc': 1},
            'index': {'tip': 8, 'pip': 6, 'mcp': 5},
            'middle': {'tip': 12, 'pip': 10, 'mcp': 9},
            'ring': {'tip': 16, 'pip': 14, 'mcp': 13},
            'pinky': {'tip': 20, 'pip': 18, 'mcp': 17}
        }
        
        # Wrist position for reference
        wrist = landmarks.landmark[0]
        
        for finger_name, indices in finger_landmarks.items():
            tip = landmarks.landmark[indices['tip']]
            pip = landmarks.landmark[indices['pip']]
            mcp = landmarks.landmark[indices['mcp']]
            
            if finger_name == 'thumb':
                # Thumb extension is measured differently (horizontal movement)
                cmc = landmarks.landmark[indices['cmc']]
                
                # Calculate thumb extension based on distance from palm
                thumb_extension = abs(tip.x - cmc.x)
                palm_center_x = (landmarks.landmark[5].x + landmarks.landmark[17].x) / 2
                
                # Check if thumb is extended away from palm
                is_extended = thumb_extension > 0.04 and abs(tip.x - palm_center_x) > 0.06
                confidence = min(1.0, thumb_extension * 15)
                
            else:
                # Other fingers: check if tip is above joints
                tip_above_pip = tip.y < pip.y - 0.01
                pip_above_mcp = pip.y < mcp.y
                
                # Additional check: finger should be pointing upward relative to wrist
                tip_above_wrist = tip.y < wrist.y + 0.05
                
                is_extended = tip_above_pip and pip_above_mcp and tip_above_wrist
                
                # Calculate confidence based on joint angles
                if is_extended:
                    extension_distance = (pip.y - tip.y) + (mcp.y - pip.y)
                    confidence = min(1.0, extension_distance * 10)
                else:
                    curl_distance = max(0, tip.y - pip.y)
                    confidence = min(1.0, curl_distance * 10)
            
            finger_states[finger_name] = is_extended
            finger_confidences[finger_name] = confidence
        
        return finger_states, finger_confidences
    
    def detect_hand_state(self, landmarks):
        """
        Detect current hand state based on finger positions
        Returns: (HandState, confidence)
        """
        finger_states, finger_confidences = self.calculate_finger_extension(landmarks)
        
        # Count extended fingers
        extended_count = sum(finger_states.values())
        avg_confidence = np.mean(list(finger_confidences.values()))
        
        # Detect specific gestures
        
        # CLOSED HAND (fist) - no fingers extended
        if extended_count == 0:
            return HandState.CLOSED, avg_confidence
        
        # OPEN HAND - most or all fingers extended
        if extended_count >= 4:
            return HandState.OPEN, avg_confidence
        
        # WRITING GESTURE - only index finger extended
        if (finger_states['index'] and 
            not finger_states['middle'] and 
            not finger_states['ring'] and 
            not finger_states['pinky']):
            # Thumb can be either extended or not for writing
            writing_confidence = (finger_confidences['index'] + 
                                (1 - finger_confidences['middle']) + 
                                (1 - finger_confidences['ring']) + 
                                (1 - finger_confidences['pinky'])) / 4
            return HandState.WRITING, writing_confidence
        
        # POINTING - index finger extended, others curled, thumb can vary
        if (finger_states['index'] and 
            not finger_states['middle'] and 
            not finger_states['ring'] and 
            not finger_states['pinky']):
            return HandState.POINTING, avg_confidence
        
        # PEACE SIGN - index and middle extended, others curled
        if (finger_states['index'] and 
            finger_states['middle'] and 
            not finger_states['ring'] and 
            not finger_states['pinky']):
            return HandState.PEACE, avg_confidence
        
        # OK SIGN - thumb and index forming circle (approximated by both extended)
        if (finger_states['thumb'] and 
            finger_states['index'] and 
            not finger_states['middle'] and 
            not finger_states['ring'] and 
            not finger_states['pinky']):
            # Additional check for thumb-index proximity
            thumb_tip = landmarks.landmark[4]
            index_tip = landmarks.landmark[8]
            distance = np.sqrt((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2)
            
            if distance < 0.05:  # Close together
                return HandState.OK, avg_confidence
        
        # Default to unknown for other combinations
        return HandState.UNKNOWN, avg_confidence * 0.5
    
    def update_state_history(self, new_state, confidence):
        """Update state history and determine stable state"""
        current_time = time.time()
        
        # Add to history
        self.state_history.append(new_state)
        self.confidence_history.append(confidence)
        
        # Check for state stability
        if len(self.state_history) >= 5:
            recent_states = list(self.state_history)[-5:]
            
            # Count occurrences of each state
            state_counts = {}
            for state in recent_states:
                state_counts[state] = state_counts.get(state, 0) + 1
            
            # Find most common state
            most_common_state = max(state_counts, key=state_counts.get)
            most_common_count = state_counts[most_common_state]
            
            # Update current state if stable and enough time has passed
            if (most_common_count >= 3 and 
                current_time - self.last_state_change > self.state_stability_threshold):
                
                if most_common_state != self.current_state:
                    print(f"🖐️ Hand state changed: {self.current_state.value} → {most_common_state.value}")
                    self.current_state = most_common_state
                    self.last_state_change = current_time
    
    def process_frame(self, frame):
        """
        Process frame and detect hand state
        Returns: (processed_frame, hand_info)
        """
        start_time = time.time()
        
        # Convert to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = self.hands.process(rgb_frame)
        
        hand_info = {
            'present': False,
            'state': HandState.UNKNOWN,
            'confidence': 0.0,
            'fingertip': None,
            'finger_states': {},
            'handedness': 'Unknown'
        }
        
        if results.multi_hand_landmarks:
            self.hand_present = True
            hand_info['present'] = True
            
            # Process first hand (assuming single hand detection)
            hand_landmarks = results.multi_hand_landmarks[0]
            self.hand_landmarks = hand_landmarks
            
            # Get handedness
            if results.multi_handedness:
                self.handedness = results.multi_handedness[0].classification[0].label
                hand_info['handedness'] = self.handedness
            
            # Detect hand state
            detected_state, confidence = self.detect_hand_state(hand_landmarks)
            
            # Update state history
            self.update_state_history(detected_state, confidence)
            
            # Update hand info
            hand_info['state'] = self.current_state
            hand_info['confidence'] = confidence
            hand_info['finger_states'] = self.finger_states
            
            # Get fingertip position (index finger)
            h, w, _ = frame.shape
            index_tip = hand_landmarks.landmark[8]
            fingertip = (int(index_tip.x * w), int(index_tip.y * h))
            hand_info['fingertip'] = fingertip
            
            # Draw hand landmarks
            self.drawing.draw_landmarks(
                frame, hand_landmarks, self.hands_module.HAND_CONNECTIONS,
                self.drawing_styles.get_default_hand_landmarks_style(),
                self.drawing_styles.get_default_hand_connections_style()
            )
            
            # Draw state information
            self.draw_state_info(frame, hand_info)
            
        else:
            # No hand detected
            self.hand_present = False
            self.hand_landmarks = None
            self.current_state = HandState.UNKNOWN
            self.state_history.clear()
            self.confidence_history.clear()
        
        # Track processing time
        processing_time = (time.time() - start_time) * 1000
        self.processing_times.append(processing_time)
        
        return frame, hand_info
    
    def draw_state_info(self, frame, hand_info):
        """Draw hand state information on frame"""
        y_offset = 30
        
        # Hand presence
        presence_text = f"Hand: {'DETECTED' if hand_info['present'] else 'NOT DETECTED'}"
        presence_color = (0, 255, 0) if hand_info['present'] else (0, 0, 255)
        cv2.putText(frame, presence_text, (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, presence_color, 2)
        
        if hand_info['present']:
            # Hand state
            state_text = f"State: {hand_info['state'].value}"
            state_color = self.get_state_color(hand_info['state'])
            cv2.putText(frame, state_text, (10, y_offset + 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, state_color, 2)
            
            # Confidence
            conf_text = f"Confidence: {hand_info['confidence']:.2f}"
            cv2.putText(frame, conf_text, (10, y_offset + 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Handedness
            hand_text = f"Hand: {hand_info['handedness']}"
            cv2.putText(frame, hand_text, (10, y_offset + 75), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Tracking status
            tracking_active = hand_info['state'] == HandState.WRITING
            tracking_text = f"Tracking: {'ACTIVE' if tracking_active else 'PAUSED'}"
            tracking_color = (0, 255, 0) if tracking_active else (255, 0, 0)
            cv2.putText(frame, tracking_text, (10, y_offset + 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, tracking_color, 2)
            
            # Draw fingertip indicator
            if hand_info['fingertip'] and tracking_active:
                fingertip = hand_info['fingertip']
                cv2.circle(frame, fingertip, 8, (0, 255, 255), -1)
                cv2.circle(frame, fingertip, 10, (255, 255, 255), 2)
    
    def get_state_color(self, state):
        """Get color for hand state"""
        color_map = {
            HandState.CLOSED: (0, 0, 255),      # Red
            HandState.OPEN: (255, 0, 0),        # Blue
            HandState.WRITING: (0, 255, 255),   # Yellow
            HandState.POINTING: (0, 255, 0),    # Green
            HandState.PEACE: (255, 0, 255),     # Magenta
            HandState.OK: (0, 255, 255),        # Cyan
            HandState.UNKNOWN: (128, 128, 128)  # Gray
        }
        return color_map.get(state, (255, 255, 255))
    
    def is_tracking_active(self):
        """Check if tracking should be active"""
        return (self.hand_present and 
                self.current_state == HandState.WRITING)
    
    def is_tracking_paused(self):
        """Check if tracking should be paused"""
        return (self.hand_present and 
                self.current_state in [HandState.OPEN, HandState.CLOSED])
    
    def get_average_processing_time(self):
        """Get average processing time"""
        return np.mean(self.processing_times) if self.processing_times else 0.0
    
    def get_current_state(self):
        """Get current hand state"""
        return self.current_state
    
    def get_state_confidence(self):
        """Get confidence of current state"""
        return np.mean(self.confidence_history) if self.confidence_history else 0.0

class HandGestureController:
    """Controller for hand gesture-based commands"""
    
    def __init__(self, hand_detector):
        self.hand_detector = hand_detector
        self.gesture_commands = {
            HandState.OPEN: self.on_hand_open,
            HandState.CLOSED: self.on_hand_closed,
            HandState.WRITING: self.on_writing_mode,
            HandState.PEACE: self.on_peace_gesture,
            HandState.OK: self.on_ok_gesture
        }
        
        self.last_command_time = {}
        self.command_cooldown = 1.0  # seconds
        
        print("✅ Hand gesture controller initialized")
    
    def process_gestures(self, hand_info):
        """Process hand gestures and execute commands"""
        if not hand_info['present']:
            return
        
        current_state = hand_info['state']
        current_time = time.time()
        
        # Check if command should be executed
        if current_state in self.gesture_commands:
            last_time = self.last_command_time.get(current_state, 0)
            
            if current_time - last_time > self.command_cooldown:
                self.gesture_commands[current_state](hand_info)
                self.last_command_time[current_state] = current_time
    
    def on_hand_open(self, hand_info):
        """Handle open hand gesture"""
        print("✋ Open hand detected - Tracking paused")
        return "PAUSE_TRACKING"
    
    def on_hand_closed(self, hand_info):
        """Handle closed hand gesture"""
        print("✊ Closed hand detected - Tracking paused")
        return "PAUSE_TRACKING"
    
    def on_writing_mode(self, hand_info):
        """Handle writing gesture"""
        print("✍️ Writing gesture detected - Tracking active")
        return "ACTIVATE_TRACKING"
    
    def on_peace_gesture(self, hand_info):
        """Handle peace gesture"""
        print("✌️ Peace gesture detected - Clear current word")
        return "CLEAR_WORD"
    
    def on_ok_gesture(self, hand_info):
        """Handle OK gesture"""
        print("👌 OK gesture detected - Complete current word")
        return "COMPLETE_WORD"

def demo_hand_detection():
    """Demo function for hand detection"""
    print("🚀 Starting Advanced Hand Detection Demo...")
    
    # Initialize components
    hand_detector = AdvancedHandDetector()
    gesture_controller = HandGestureController(hand_detector)
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open camera")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    print("📋 Instructions:")
    print("   • Show different hand gestures to see detection")
    print("   • OPEN hand (all fingers) = Pause tracking")
    print("   • CLOSED hand (fist) = Pause tracking")
    print("   • INDEX finger only = Writing mode")
    print("   • PEACE sign = Clear command")
    print("   • OK sign = Complete command")
    print("   • Press ESC to exit")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Flip for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Process hand detection
            processed_frame, hand_info = hand_detector.process_frame(frame)
            
            # Process gestures
            gesture_controller.process_gestures(hand_info)
            
            # Add performance info
            avg_time = hand_detector.get_average_processing_time()
            cv2.putText(processed_frame, f"Processing: {avg_time:.1f}ms", 
                       (10, processed_frame.shape[0] - 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Show frame
            cv2.imshow('Advanced Hand Detection Demo', processed_frame)
            
            # Handle input
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
    
    except KeyboardInterrupt:
        print("\n👋 Demo interrupted by user")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("✅ Demo finished")

if __name__ == "__main__":
    demo_hand_detection()