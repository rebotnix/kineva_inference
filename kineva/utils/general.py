import numpy as np
import cv2

def generate_color_palette(classes, seed=42):
    """Generate a consistent color palette for classes using a seed"""
    np.random.seed(seed)
    colors = {}
    golden_ratio = 0.618033988749895  # Use golden ratio for better color distribution
    num_classes = len(classes)
    for i, el in enumerate(classes):
        # Generate HSV values
        hue = (i * golden_ratio) % 1.0
        saturation = 0.8 + np.random.random() * 0.2  # 0.8-1.0
        value = 0.8 + np.random.random() * 0.2  # 0.8-1.0
        
        # Convert HSV to RGB
        h = hue * 6
        c = value * saturation
        x = c * (1 - abs(h % 2 - 1))
        m = value - c
        
        if h < 1:
            rgb = (c, x, 0)
        elif h < 2:
            rgb = (x, c, 0)
        elif h < 3:
            rgb = (0, c, x)
        elif h < 4:
            rgb = (0, x, c)
        elif h < 5:
            rgb = (x, 0, c)
        else:
            rgb = (c, 0, x)
            
        # Convert to 8-bit RGB values
        color = tuple(int((cc + m) * 255) for cc in rgb)
        colors[el] = color
    
    return colors

def draw_dashed_rectangle(img, pt1, pt2, color=(0,0,255), thickness=2, dash_length=10):
    """
    Draw a dashed rectangle from pt1 to pt2.
    
    :param img: Image to draw on
    :param pt1: Top-left corner of the rectangle (x, y)
    :param pt2: Bottom-right corner of the rectangle (x, y)
    :param color: Line color (B, G, R)
    :param thickness: Line thickness
    :param dash_length: Length of dashes
    """
    x1, y1 = pt1
    x2, y2 = pt2

    # Draw top side
    for x in range(x1, x2, dash_length * 2):
        cv2.line(img, (x, y1), (min(x + dash_length, x2), y1), color, thickness)
    
    # Draw bottom side
    for x in range(x1, x2, dash_length * 2):
        cv2.line(img, (x, y2), (min(x + dash_length, x2), y2), color, thickness)
    
    # Draw left side
    for y in range(y1, y2, dash_length * 2):
        cv2.line(img, (x1, y), (x1, min(y + dash_length, y2)), color, thickness)
    
    # Draw right side
    for y in range(y1, y2, dash_length * 2):
        cv2.line(img, (x2, y), (x2, min(y + dash_length, y2)), color, thickness)

    return img
