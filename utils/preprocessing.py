import numpy as np
import cv2

def draw_path_on_blank(path, img_size=256):
    blank = np.ones((img_size, img_size), dtype=np.uint8) * 255
    for i in range(1, len(path)):
        cv2.line(blank, path[i-1], path[i], (0), 4)
    return blank
