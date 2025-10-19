#!/usr/bin/env python3
"""
Simple Air Writing Demo - Works with or without MediaPipe
Basic hand tracking and path visualization for testing setup
"""

import cv2
import numpy as np
import time
from collections import deque

class SimpleHandTracker:
    def __init__(self):
        self.mp_available = False
        self.hands = None
        self.hands_module = None
        self.drawing = None
        
        # Try to import MediaPipe
        try:
            import mediapipe as mp
            self.hands_module = mp.solutions.hands
            self.hands = self.hands_module.Hands(
                max_num_hands=1,
                min_detection_confidence=0.5,  # Lowered for better detection
                min_tracking_confidence=0.3    # Lowered for smoother tracking
            )
            self.drawing = mp.solutions.drawing_utils
            self.mp_available = True
            print("✅ MediaPipe loaded successfully")
        except ImportError:
            print("⚠️  MediaPipe not available - using fallback color tracking")
            print("   For better results, install MediaPipe:")
            print("   pip install mediapipe")
        except Exception as e:
            print(f"⚠️  MediaPipe error: {e}")
            print("   Using fallback color tracking")
        
        self.trail = deque(maxlen=150)  # Longer trail
        self.last_point = None
        self.velocity = 0.0
    
    def get_fingertip_basic(self, frame):
        """Basic fingertip detection using color/contour detection"""
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Define range for skin color (adjust as needed)
        lower_skin = np.array([0, 20, 70])
        upper_skin = np.array([20, 255, 255])
        
        # Create mask
        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Get largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            
            if cv2.contourArea(largest_contour) > 1000:
                # Get the topmost point (fingertip)
                topmost = tuple(largest_contour[largest_contour[:, :, 1].argmin()][0])
                return topmost
        
        return None
    
    def get_fingertip_mediapipe(self, frame):
        """MediaPipe-based fingertip detection"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks
                self.drawing.draw_landmarks(
                    frame, hand_landmarks, self.hands_module.HAND_CONNECTIONS
                )
                
                # Get index fingertip (landmark 8)
                h, w, _ = frame.shape
                fingertip = hand_landmarks.landmark[8]
                x = int(fingertip.x * w)
                y = int(fingertip.y * h)
                
                return (x, y)
        
        return None
    
    def get_fingertip(self, frame):
        """Get fingertip position using available method"""
        if self.mp_available:
            return self.get_fingertip_mediapipe(frame)
        else:
            return self.get_fingertip_basic(frame)
    
    def update_trail(self, point):
        """Update the drawing trail"""
        if point:
            self.trail.append(point)
            self.last_point = point
    
    def draw_trail(self, frame):
        """Draw the trail on the frame"""
        if len(self.trail) < 2:
            return
        
        # Draw trail with gradient
        for i in range(1, len(self.trail)):
            alpha = i / len(self.trail)
            color = (int(255 * (1-alpha)), int(255 * alpha), 0)  # Blue to red gradient
            thickness = max(1, int(5 * alpha))
            cv2.line(frame, self.trail[i-1], self.trail[i], color, thickness)

class SimpleAirWritingDemo:
    def __init__(self):
        self.tracker = SimpleHandTracker()
        self.cap = cv2.VideoCapture(0)
        
        if not self.cap.isOpened():
            print("❌ Could not open camera")
            return
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        self.writing_mode = False
        self.current_path = []
        self.letters = []
        
        print("✅ Simple Air Writing Demo initialized")
        print("📋 Instructions:")
        print("   - Hold up your index finger to start writing")
        print("   - Move your finger to draw in the air")
        print("   - Press SPACE to save current letter")
        print("   - Press C to clear")
        print("   - Press ESC to exit")
    
    def draw_ui(self, frame):
        """Draw user interface"""
        h, w = frame.shape[:2]
        
        # Draw semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (w-10, 100), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Status text
        status = "WRITING" if self.writing_mode else "READY"
        color = (0, 255, 0) if self.writing_mode else (0, 255, 255)
        cv2.putText(frame, f"Status: {status}", (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        # Letter count
        cv2.putText(frame, f"Letters drawn: {len(self.letters)}", (20, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Instructions
        instructions = [
            "SPACE: Save letter | C: Clear | ESC: Exit",
            "Hold up index finger and draw letters in air"
        ]
        
        for i, instruction in enumerate(instructions):
            cv2.putText(frame, instruction, (20, h - 60 + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    def run(self):
        """Run the demo"""
        print("\n🚀 Starting Simple Air Writing Demo...")
        print("Position yourself 2-3 feet from the camera")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("❌ Failed to read from camera")
                break
            
            # Flip frame horizontally
            frame = cv2.flip(frame, 1)
            
            # Get fingertip position
            fingertip = self.tracker.get_fingertip(frame)
            
            if fingertip:
                # Draw fingertip
                cv2.circle(frame, fingertip, 10, (0, 255, 0), -1)
                cv2.circle(frame, fingertip, 12, (255, 255, 255), 2)
                
                # Update trail
                self.tracker.update_trail(fingertip)
                self.current_path.append(fingertip)
                self.writing_mode = True
            else:
                self.writing_mode = False
            
            # Draw trail
            self.tracker.draw_trail(frame)
            
            # Draw UI
            self.draw_ui(frame)
            
            # Show frame
            cv2.imshow("Simple Air Writing Demo", frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' '):  # Save current letter
                if len(self.current_path) > 5:
                    self.letters.append(self.current_path.copy())
                    print(f"Letter {len(self.letters)} saved with {len(self.current_path)} points")
                self.current_path.clear()
                self.tracker.trail.clear()
            
            elif key == ord('c'):  # Clear
                self.current_path.clear()
                self.tracker.trail.clear()
                self.letters.clear()
                print("Cleared all data")
            
            elif key == 27:  # ESC
                break
        
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()
        
        print(f"\n📊 Session Summary:")
        print(f"   Letters drawn: {len(self.letters)}")
        print(f"   Total points: {sum(len(letter) for letter in self.letters)}")
        print("👋 Demo completed!")

def test_camera():
    """Test camera functionality"""
    print("📹 Testing camera...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Camera not accessible")
        return False
    
    ret, frame = cap.read()
    if ret and frame is not None:
        print(f"✅ Camera working - Frame size: {frame.shape}")
        cap.release()
        return True
    else:
        print("❌ Camera not providing frames")
        cap.release()
        return False

def test_mediapipe():
    """Test MediaPipe functionality"""
    print("🖐️  Testing MediaPipe...")
    try:
        import mediapipe as mp
        hands = mp.solutions.hands.Hands()
        print("✅ MediaPipe hands initialized successfully")
        return True
    except ImportError:
        print("❌ MediaPipe not available")
        return False
    except Exception as e:
        print(f"❌ MediaPipe error: {e}")
        return False

def main():
    """Main function"""
    print("🖐️  SIMPLE AIR WRITING DEMO")
    print("=" * 40)
    
    # Run tests
    camera_ok = test_camera()
    mediapipe_ok = test_mediapipe()
    
    if not camera_ok:
        print("❌ Cannot proceed without camera")
        return
    
    if not mediapipe_ok:
        print("⚠️  MediaPipe not available - using basic tracking")
        print("   Install MediaPipe for better results: pip install mediapipe")
    
    # Run demo
    try:
        demo = SimpleAirWritingDemo()
        demo.run()
    except KeyboardInterrupt:
        print("\n👋 Demo interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()