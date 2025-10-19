import numpy as np
import cv2
from scipy import ndimage
from scipy.spatial.distance import euclidean

def normalize_path(path):
    """Normalize path coordinates to a standard range"""
    if len(path) < 2:
        return path
    
    path_array = np.array(path)
    
    # Find bounding box
    min_x, min_y = np.min(path_array, axis=0)
    max_x, max_y = np.max(path_array, axis=0)
    
    # Calculate dimensions
    width = max_x - min_x
    height = max_y - min_y
    
    if width == 0 or height == 0:
        return path
    
    # Normalize to 200x200 with padding
    target_size = 200
    padding = 28
    
    # Scale factor to fit in target size with padding
    scale = min((target_size - 2 * padding) / width, 
                (target_size - 2 * padding) / height)
    
    # Normalize coordinates
    normalized_path = []
    for x, y in path:
        norm_x = int((x - min_x) * scale + padding)
        norm_y = int((y - min_y) * scale + padding)
        normalized_path.append((norm_x, norm_y))
    
    return normalized_path

def smooth_path(path, window_size=3):
    """Apply smoothing to reduce noise in the path"""
    if len(path) < window_size:
        return path
    
    path_array = np.array(path)
    smoothed = np.zeros_like(path_array)
    
    for i in range(len(path)):
        start_idx = max(0, i - window_size // 2)
        end_idx = min(len(path), i + window_size // 2 + 1)
        smoothed[i] = np.mean(path_array[start_idx:end_idx], axis=0)
    
    return [(int(x), int(y)) for x, y in smoothed]

def calculate_path_features(path):
    """Calculate various features of the drawn path"""
    if len(path) < 2:
        return {}
    
    path_array = np.array(path)
    
    # Basic metrics
    total_length = sum(euclidean(path[i], path[i+1]) for i in range(len(path)-1))
    
    # Bounding box
    min_x, min_y = np.min(path_array, axis=0)
    max_x, max_y = np.max(path_array, axis=0)
    width = max_x - min_x
    height = max_y - min_y
    aspect_ratio = width / height if height > 0 else 1.0
    
    # Direction changes (corners/curves)
    direction_changes = 0
    if len(path) > 2:
        for i in range(1, len(path) - 1):
            v1 = np.array(path[i]) - np.array(path[i-1])
            v2 = np.array(path[i+1]) - np.array(path[i])
            
            # Calculate angle between vectors
            if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                cos_angle = np.clip(cos_angle, -1, 1)
                angle = np.arccos(cos_angle)
                
                if angle > np.pi / 4:  # 45 degrees threshold
                    direction_changes += 1
    
    return {
        'length': total_length,
        'width': width,
        'height': height,
        'aspect_ratio': aspect_ratio,
        'direction_changes': direction_changes,
        'num_points': len(path)
    }

def draw_path_on_blank(path, img_size=256):
    """Enhanced path drawing with normalization and smoothing"""
    if len(path) < 2:
        return np.ones((img_size, img_size), dtype=np.uint8) * 255
    
    # Normalize and smooth the path
    normalized_path = normalize_path(path)
    smoothed_path = smooth_path(normalized_path)
    
    # Create blank image
    blank = np.ones((img_size, img_size), dtype=np.uint8) * 255
    
    # Draw the path with variable thickness based on speed
    for i in range(1, len(smoothed_path)):
        # Calculate thickness based on distance (speed indicator)
        distance = euclidean(smoothed_path[i-1], smoothed_path[i])
        thickness = max(2, min(6, int(8 - distance / 5)))
        
        cv2.line(blank, smoothed_path[i-1], smoothed_path[i], (0), thickness)
    
    # Apply slight blur to smooth edges
    blank = cv2.GaussianBlur(blank, (3, 3), 0)
    
    return blank

def enhance_letter_image(image):
    """Apply image enhancement techniques for better recognition"""
    # Apply morphological operations to clean up the image
    kernel = np.ones((2, 2), np.uint8)
    
    # Closing to fill small gaps
    enhanced = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    
    # Opening to remove noise
    enhanced = cv2.morphologyEx(enhanced, cv2.MORPH_OPEN, kernel)
    
    return enhanced

def extract_letter_region(image, padding=4):
    """Extract the minimal bounding box containing the letter"""
    # Find contours
    contours, _ = cv2.findContours(255 - image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return image
    
    # Get bounding box of largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # Add padding
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(image.shape[1] - x, w + 2 * padding)
    h = min(image.shape[0] - y, h + 2 * padding)
    
    # Extract region
    letter_region = image[y:y+h, x:x+w]
    
    # Resize to square maintaining aspect ratio
    size = max(w, h)
    square_image = np.ones((size, size), dtype=np.uint8) * 255
    
    # Center the letter in the square
    start_x = (size - w) // 2
    start_y = (size - h) // 2
    square_image[start_y:start_y+h, start_x:start_x+w] = letter_region
    
    return square_image
