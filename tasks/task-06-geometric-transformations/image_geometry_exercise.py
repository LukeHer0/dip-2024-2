# image_geometry_exercise.py
# STUDENT'S EXERCISE FILE

"""
Exercise:
Implement a function `apply_geometric_transformations(img)` that receives a grayscale image
represented as a NumPy array (2D array) and returns a dictionary with the following transformations:

1. Translated image (shift right and down)
2. Rotated image (90 degrees clockwise)
3. Horizontally stretched image (scale width by 1.5)
4. Horizontally mirrored image (flip along vertical axis)
5. Barrel distorted image (simple distortion using a radial function)

You must use only NumPy to implement these transformations. Do NOT use OpenCV, PIL, skimage or similar libraries.

Function signature:
    def apply_geometric_transformations(img: np.ndarray) -> dict:

The return value should be like:
{
    "translated": np.ndarray,
    "rotated": np.ndarray,
    "stretched": np.ndarray,
    "mirrored": np.ndarray,
    "distorted": np.ndarray
}
"""

import numpy as np
import matplotlib as plt
import matplotlib.pyplot as plt

def apply_geometric_transformations(img: np.ndarray) -> dict:
    h, w = img.shape
    fill = 0

    translated_img = np.full_like(img, fill)

    translated_img[1:, 1:] = img[:-1, :-1]

    rotated_img = np.rot90(img, k=-1)

    scale = 1.5
    scaled_h, scaled_w = int(h * scale), int(w * scale)
    stretched_img = np.zeros((scaled_h, scaled_w), dtype=img.dtype)

    y_map = np.arange(scaled_h) / scale
    x_map = np.arange(scaled_w) / scale

    y0 = np.floor(y_map).astype(int)
    x0 = np.floor(x_map).astype(int)

    y1 = np.clip(y0 + 1, 0, h - 1)
    x1 = np.clip(x0 + 1, 0, w - 1)

    wy = y_map - y0
    wx = x_map - x0

    A = img[y0[:, None], x0]
    B = img[y0[:, None], x1]
    C = img[y1[:, None], x0]
    D = img[y1[:, None], x1]

    stretched_img = (A * (1 - wx) + B * wx) * (1 - wy)[:, None] + (C * (1 - wx) + D * wx) * wy[:, None]

    mirrored_img = img[::-1,:]

    distorted_img = np.full_like(img, fill)

    Y, X = np.indices((h, w))
    cx, cy = w // 2, h // 2
 
    X_norm = (X - cx) / cx
    Y_norm = (Y - cy) / cy
    
    r = np.sqrt(X_norm**2 + Y_norm**2)
    theta = np.arctan2(Y_norm, X_norm)
    
    r_distorted = r + 0.05 * (r ** 3)
    
    Xd_norm = r_distorted * np.cos(theta)
    Yd_norm = r_distorted * np.sin(theta)
    
    x_new = (Xd_norm * cx + cx).astype(int)
    y_new = (Yd_norm * cy + cy).astype(int)
    
    valid_mask = (x_new >= 0) & (x_new < w) & (y_new >= 0) & (y_new < h)
    
    distorted_img[valid_mask] = img[y_new[valid_mask], x_new[valid_mask]]

    return {"translated": translated_img,
            "rotated": rotated_img,
            "stretched": stretched_img,
            "mirrored": mirrored_img,
            "distorted": distorted_img}
    pass