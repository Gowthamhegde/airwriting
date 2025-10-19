#!/usr/bin/env python3
"""
Hand Tracking Only System
Optimized hand movement tracking without letter recognition
"""

import cv2
import numpy as np
import time
from collections import deque
import threading

class OptimizedHandTracker:
    """Optimized hand tracker focused on movement detection"""
    
    def __init__(self):
        self.trail_points = deque(maxlen=50)
        self.last_position = None
        self.movement_threshold = 15
        self.smoothing_factor = 0.7
        self.hand_detected = False
        
        # Color detection parameters
        self.lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        self.upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        
        # Movement tracking
        self.velocity_history = deque(maxlen=10)
        self.is_writing = False
        self.writing_threshold = 5
        
    def detect_hand_contour(self, frame):
        """Detect hand using color and contour detection"""
        # Convert to HSV for better skin detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create skin mask
        mask = cv2.inRange(hsv, self.lower_skin, self.upper_skin)
        
        # Morphological operations to clean up mask
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find largest contour (likely the hand)
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Filter by area
            if cv2.contourArea(largest_contour) > 1000:
                return largest_contour, mask
        
        return None, mask
    
    def get_fingertip_position(self, contour):
        """Get fingertip position from hand contour"""
        # Find convex hull
        hull = cv2.convexHull(contour)
        
        # Find the topmost point (likely fingertip)
        topmost = tuple(hull[hull[:, :, 1].argmin()][0])
        
        return topmost
    
    def smooth_position(self, new_pos):
        """Apply smoothing to reduce jitter"""
        if self.last_position is None:
            self.last_position = new_pos
            return new_pos
        
        # Exponential smoothing
        smoothed_x = int(self.smoothing_factor * new_pos[0] + (1 - self.smoothing_factor) * self.last_position[0])
        smoothed_y = int(self.smoothing_factor * new_pos[1] + (1 - self.smoothing_factor) * self.last_position[1])
        
        smoothed_pos = (smoothed_x, smoothed_y)
        self.last_position = smoothed_pos
        return smoothed_pos
    
    def calculate_velocity(self, current_pos):
        """Calculate movement velocity"""
        if len(self.trail_points) < 2:
            return 0
        
        prev_pos = self.trail_points[-1]
        distance = np.sqrt((current_pos[0] - prev_pos[0])**2 + (current_pos[1] - prev_pos[1])**2)
        
        self.velocity_history.append(distance)
        avg_velocity = np.mean(self.velocity_history)
        
        # Determine if writing motion
        self.is_writing = avg_velocity > self.writing_threshold
        
        return avg_velocity
    
    def process_frame(self, frame):
        """Process frame and detect hand movement"""
        # Detect hand
        contour, mask = self.detect_hand_contour(frame)
        
        if contour is not None:
            self.hand_detected = True
            
            # Get fingertip position
            fingertip = self.get_fingertip_position(contour)
            
            # Smooth the position
            smooth_pos = self.smooth_position(fingertip)
            
            # Calculate velocity
            velocity = self.calculate_velocity(smooth_pos)
            
            # Add to trail if moving significantly
            if (not self.trail_points or 
                np.sqrt((smooth_pos[0] - self.trail_points[-1][0])**2 + 
                       (smooth_pos[1] - self.trail_points[-1][1])**2) > self.movement_threshold):
                self.trail_points.append(smooth_pos)
            
            return smooth_pos, velocity, contour, mask
        else:
            self.hand_detected = False
            return None, 0, None, mask

class HandTrackingSystem:
    """Main hand tracking system"""
    
    def __init__(self):
        self.tracker = OptimizedHandTracker()
        self.cap = None
        self.running = False
        self.show_trail = True
        self.show_mask = False
        self.show_contour = True
        
        # Display settings
        self.trail_color = (0, 255, 0)  # Green
        self.fingertip_color = (0, 0, 255)  # Red
        self.contour_color = (255, 0, 0)  # Blue
        
    def initialize_camera(self):
        """Initialize camera"""
        print("🎥 Initializing camera...")
        self.cap = cv2.VideoCapture(0)
        
        if not self.cap.isOpened():
            print("❌ Could not open camera")
            return False
        
        # Set camera properties for better performance
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("✅ Camera initialized")
        return True
    
    def draw_trail(self, frame):
        """Draw movement trail"""
        if not self.show_trail or len(self.tracker.trail_points) < 2:
            return
        
        # Draw trail with fading effect
        for i in range(1, len(self.tracker.trail_points)):
            # Calculate alpha based on position in trail
            alpha = i / len(self.tracker.trail_points)
            thickness = max(1, int(alpha * 5))
            
            cv2.line(frame, 
                    self.tracker.trail_points[i-1], 
                    self.tracker.trail_points[i], 
                    self.trail_color, 
                    thickness)
    
    def draw_info(self, frame, position, velocity):
        """Draw information overlay"""
        height, width = frame.shape[:2]
        
        # Status info
        status_text = "✅ Hand Detected" if self.tracker.hand_detected else "❌ No Hand"
        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if self.tracker.hand_detected else (0, 0, 255), 2)
        
        if position:
            # Position info
            pos_text = f"Position: ({position[0]}, {position[1]})"
            cv2.putText(frame, pos_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Velocity info
            vel_text = f"Velocity: {velocity:.1f}"
            cv2.putText(frame, vel_text, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Writing status
            writing_text = "✍️ Writing" if self.tracker.is_writing else "✋ Stationary"
            cv2.putText(frame, writing_text, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255) if self.tracker.is_writing else (255, 255, 255), 1)
        
        # Controls
        controls = [
            "Controls:",
            "T - Toggle trail",
            "M - Toggle mask",
            "C - Toggle contour", 
            "R - Reset trail",
            "ESC - Exit"
        ]
        
        for i, control in enumerate(controls):
            cv2.putText(frame, control, (width - 200, 30 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    def run(self):
        """Run the hand tracking system"""
        if not self.initialize_camera():
            return
        
        print("\n🖐️  HAND TRACKING SYSTEM")
        print("=" * 50)
        print("📋 Instructions:")
        print("   • Hold your hand in front of the camera")
        print("   • Move your finger to see tracking")
        print("   • Green trail shows movement path")
        print("   • Red dot shows fingertip position")
        print("\n⌨️  Controls:")
        print("   T - Toggle trail display")
        print("   M - Toggle mask view")
        print("   C - Toggle contour display")
        print("   R - Reset trail")
        print("   ESC - Exit")
        print("=" * 50)
        
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
                
                # Process frame
                position, velocity, contour, mask = self.tracker.process_frame(frame)
                
                # Create display frame
                display_frame = frame.copy()
                
                # Draw trail
                self.draw_trail(display_frame)
                
                # Draw contour if available
                if contour is not None and self.show_contour:
                    cv2.drawContours(display_frame, [contour], -1, self.contour_color, 2)
                
                # Draw fingertip position
                if position:
                    cv2.circle(display_frame, position, 8, self.fingertip_color, -1)
                    cv2.circle(display_frame, position, 12, (255, 255, 255), 2)
                
                # Draw info overlay
                self.draw_info(display_frame, position, velocity)
                
                # Show frames
                cv2.imshow('Hand Tracking System', display_frame)
                
                if self.show_mask:
                    cv2.imshow('Hand Mask', mask)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == 27:  # ESC
                    break
                elif key == ord('t') or key == ord('T'):
                    self.show_trail = not self.show_trail
                    print(f"Trail display: {'ON' if self.show_trail else 'OFF'}")
                elif key == ord('m') or key == ord('M'):
                    self.show_mask = not self.show_mask
                    if not self.show_mask:
                        cv2.destroyWindow('Hand Mask')
                    print(f"Mask display: {'ON' if self.show_mask else 'OFF'}")
                elif key == ord('c') or key == ord('C'):
                    self.show_contour = not self.show_contour
                    print(f"Contour display: {'ON' if self.show_contour else 'OFF'}")
                elif key == ord('r') or key == ord('R'):
                    self.tracker.trail_points.clear()
                    print("Trail reset")
                
                # Calculate FPS
                fps_counter += 1
                if fps_counter % 30 == 0:
                    fps = 30 / (time.time() - fps_start_time)
                    print(f"📊 FPS: {fps:.1f}")
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
    print("🚀 Starting Hand Tracking System...")
    
    system = HandTrackingSystem()
    system.run()
    
    print("👋 Hand Tracking System stopped")

if __name__ == "__main__":
    main()