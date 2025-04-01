# image_similarity_exercise.py
# STUDENT'S EXERCISE FILE

"""
Exercise:
Implement a function `compare_images(i1, i2)` that receives two grayscale images
represented as NumPy arrays (2D arrays of shape (H, W)) and returns a dictionary with the following metrics:

1. Mean Squared Error (MSE)
2. Peak Signal-to-Noise Ratio (PSNR)
3. Structural Similarity Index (SSIM) - simplified version without using external libraries
4. Normalized Pearson Correlation Coefficient (NPCC)

You must implement these functions yourself using only NumPy (no OpenCV, skimage, etc).

Each function should be implemented as a helper function and called inside `compare_images(i1, i2)`.

Function signature:
    def compare_images(i1: np.ndarray, i2: np.ndarray) -> dict:

The return value should be like:
{
    "mse": float,
    "psnr": float,
    "ssim": float,
    "npcc": float
}

Assume that i1 and i2 are normalized grayscale images (values between 0 and 1).
"""

import numpy as np

def compare_images(i1: np.ndarray, i2: np.ndarray) -> dict:
    if i1.shape != i2.shape:
        raise ValueError("Imagens de tamanho incompatível")
    
    mse = np.mean((i1 - i2) ** 2)

    max = 255.0
    
    if mse == 0:
        psnr = float('inf')
    else:
        psnr = 20 * np.log10(max) - 10 * np.log10(mse)

    c1 = (0.01 * max) ** 2
    c2 = (0.03 * max) ** 2
    c3 = c2/2
    pixel_mean1 = np.mean(i1)
    pixel_mean2 = np.mean(i2)
    variance1 = np.var(i1)
    variance2 = np.var(i2)
    covariance = np.cov(i1.flatten(), i2.flatten())[0,1]
    
    luminance = (2 * pixel_mean1 * pixel_mean2 + c1)/(pixel_mean1 ** 2 + pixel_mean2 ** 2 + c1)

    contrast = (2 * variance1 * variance2 + c2)/(variance1 **2 + variance2 ** 2 + c2)

    structure = (covariance + c3)/(np.sqrt(variance1 * variance2) + c3)

    ssim = luminance * contrast * structure

    npcc = np.corrcoef(i1.flatten(), i2.flatten())[0,1]

    return {"mse": mse,
            "psnr": psnr, 
            "ssim": ssim,
            "npcc": npcc} 

    pass