import numpy as np
import cv2
from scipy import ndimage
from scipy.spatial.distance import euclidean
from scipy.signal import savgol_filter
import time

def normalize_path(path, target_size=32, padding=4):
    """Enhanced path normalization optimized for air writing"""
    if len(path) < 2:
        return path

    path_array = np.array(path, dtype=np.float32)

    # Remove outliers using IQR method
    if len(path_array) > 10:
        x_coords = path_array[:, 0]
        y_coords = path_array[:, 1]

        # Calculate IQR for outlier removal
        x_q75, x_q25 = np.percentile(x_coords, [75, 25])
        y_q75, y_q25 = np.percentile(y_coords, [75, 25])
        x_iqr = x_q75 - x_q25
        y_iqr = y_q75 - y_q25

        # Filter outliers
        x_mask = (x_coords >= x_q25 - 1.5 * x_iqr) & (x_coords <= x_q75 + 1.5 * x_iqr)
        y_mask = (y_coords >= y_q25 - 1.5 * y_iqr) & (y_coords <= y_q75 + 1.5 * y_iqr)
        valid_mask = x_mask & y_mask

        if np.sum(valid_mask) > 2:
            path_array = path_array[valid_mask]

    # Find bounding box
    min_x, min_y = np.min(path_array, axis=0)
    max_x, max_y = np.max(path_array, axis=0)

    width = max_x - min_x
    height = max_y - min_y

    if width <= 0 or height <= 0:
        return path

    # Calculate scale to fit in target size with padding
    scale = min((target_size - 2 * padding) / width,
                (target_size - 2 * padding) / height)

    # Center the path
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    offset_x = target_size / 2 - center_x * scale
    offset_y = target_size / 2 - center_y * scale

    # Normalize coordinates
    normalized_path = []
    for x, y in path_array:
        norm_x = int(x * scale + offset_x)
        norm_y = int(y * scale + offset_y)
        # Clamp to image bounds
        norm_x = np.clip(norm_x, 0, target_size - 1)
        norm_y = np.clip(norm_y, 0, target_size - 1)
        normalized_path.append((norm_x, norm_y))

    return normalized_path

def smooth_path(path, window_size=5, method='savgol'):
    """Advanced path smoothing with multiple methods"""
    if len(path) < window_size:
        return path

    path_array = np.array(path, dtype=np.float32)

    if method == 'savgol' and len(path_array) > window_size:
        try:
            # Savitzky-Golay filter for smooth curves
            smoothed_x = savgol_filter(path_array[:, 0], window_size, 3)
            smoothed_y = savgol_filter(path_array[:, 1], window_size, 3)
            smoothed = np.column_stack((smoothed_x, smoothed_y))
        except:
            # Fallback to moving average
            smoothed = np.zeros_like(path_array)
            for i in range(len(path_array)):
                start_idx = max(0, i - window_size // 2)
                end_idx = min(len(path_array), i + window_size // 2 + 1)
                smoothed[i] = np.mean(path_array[start_idx:end_idx], axis=0)
    else:
        # Moving average smoothing
        smoothed = np.zeros_like(path_array)
        for i in range(len(path_array)):
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(path_array), i + window_size // 2 + 1)
            smoothed[i] = np.mean(path_array[start_idx:end_idx], axis=0)

    return [(int(x), int(y)) for x, y in smoothed]

def calculate_path_features(path):
    """Enhanced feature calculation for air writing patterns"""
    if len(path) < 2:
        return {}

    path_array = np.array(path, dtype=np.float32)

    # Basic metrics
    total_length = sum(euclidean(path[i], path[i+1]) for i in range(len(path)-1))

    # Bounding box
    min_x, min_y = np.min(path_array, axis=0)
    max_x, max_y = np.max(path_array, axis=0)
    width = max_x - min_x
    height = max_y - min_y
    aspect_ratio = width / height if height > 0 else 1.0

    # Stroke analysis
    stroke_speeds = []
    stroke_curvatures = []
    stroke_angles = []

    for i in range(1, len(path_array) - 1):
        # Speed calculation
        v1 = path_array[i] - path_array[i-1]
        v2 = path_array[i+1] - path_array[i]
        speed1 = np.linalg.norm(v1)
        speed2 = np.linalg.norm(v2)
        avg_speed = (speed1 + speed2) / 2
        stroke_speeds.append(avg_speed)

        # Angle calculation
        if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            cos_angle = np.clip(cos_angle, -1, 1)
            angle = np.arccos(cos_angle)
            stroke_angles.append(angle)

            # Curvature (change in direction)
            curvature = angle / max(speed1 + speed2, 1e-6)
            stroke_curvatures.append(curvature)

    # Statistical features
    features = {
        'length': total_length,
        'width': width,
        'height': height,
        'aspect_ratio': aspect_ratio,
        'num_points': len(path),
        'avg_speed': np.mean(stroke_speeds) if stroke_speeds else 0,
        'speed_variance': np.var(stroke_speeds) if stroke_speeds else 0,
        'avg_curvature': np.mean(stroke_curvatures) if stroke_curvatures else 0,
        'curvature_variance': np.var(stroke_curvatures) if stroke_curvatures else 0,
        'direction_changes': sum(1 for angle in stroke_angles if angle > np.pi / 4),
        'total_angle_change': sum(stroke_angles) if stroke_angles else 0
    }

    return features

def draw_path_on_blank(path, img_size=32, stroke_variation=True):
    """Optimized path drawing for real-time processing"""
    if len(path) < 2:
        return np.ones((img_size, img_size), dtype=np.uint8) * 255

    # Normalize path to target size
    normalized_path = normalize_path(path, img_size, padding=2)
    smoothed_path = smooth_path(normalized_path, window_size=3)

    # Create blank image
    blank = np.ones((img_size, img_size), dtype=np.uint8) * 255

    # Draw path with optimized thickness
    for i in range(1, len(smoothed_path)):
        p1 = smoothed_path[i-1]
        p2 = smoothed_path[i]

        # Calculate distance for thickness
        distance = euclidean(p1, p2)

        if stroke_variation:
            # Variable thickness based on speed simulation
            thickness = max(1, min(4, int(6 - distance)))
        else:
            thickness = 2

        cv2.line(blank, p1, p2, (0), thickness)

    # Light blur for smoothing (optimized for speed)
    blank = cv2.GaussianBlur(blank, (3, 3), 0.5)

    return blank

def enhance_letter_image(image):
    """Enhanced image processing for air writing letters"""
    # Convert to binary with adaptive threshold
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Adaptive thresholding for better letter extraction
    binary = cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )

    # Morphological operations to clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))

    # Close small gaps
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # Remove small noise
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # Fill holes
    binary = ndimage.binary_fill_holes(binary).astype(np.uint8) * 255

    return binary

def extract_letter_region(image, padding=2):
    """Optimized letter region extraction"""
    # Find contours
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return image

    # Get largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Skip if too small
    if w < 5 or h < 5:
        return image

    # Add minimal padding
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(image.shape[1] - x, w + 2 * padding)
    h = min(image.shape[0] - y, h + 2 * padding)

    # Extract region
    letter_region = image[y:y+h, x:x+w]

    # Resize to square (32x32 for optimized model)
    size = 32
    square_image = np.ones((size, size), dtype=np.uint8) * 255

    # Scale to fit
    scale = min(size / w, size / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    # Resize letter region
    resized = cv2.resize(letter_region, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Center in square
    start_x = (size - new_w) // 2
    start_y = (size - new_h) // 2
    square_image[start_y:start_y+new_h, start_x:start_x+new_w] = resized

    return square_image

def preprocess_for_recognition(path, img_size=32):
    """Complete preprocessing pipeline for letter recognition"""
    start_time = time.time()

    # Draw path to image
    image = draw_path_on_blank(path, img_size)

    # Enhance image
    enhanced = enhance_letter_image(image)

    # Extract letter region
    final_image = extract_letter_region(enhanced)

    # Normalize to 0-1
    final_image = final_image.astype(np.float32) / 255.0

    # Add channel dimension
    final_image = np.expand_dims(final_image, axis=[0, -1])

    processing_time = time.time() - start_time

    return final_image, processing_time

def batch_preprocess_paths(paths, img_size=32):
    """Batch preprocessing for multiple paths"""
    processed_images = []
    processing_times = []

    for path in paths:
        img, proc_time = preprocess_for_recognition(path, img_size)
        processed_images.append(img[0])  # Remove batch dimension
        processing_times.append(proc_time)

    # Stack into batch
    batch = np.stack(processed_images, axis=0)

    return batch, processing_times
