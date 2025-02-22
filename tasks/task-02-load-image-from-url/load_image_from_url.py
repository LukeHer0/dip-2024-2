import argparse
import numpy as np
import cv2 as cv

def load_image_from_url(url, **kwargs):
    """
    Loads an image from an Internet URL with optional arguments for OpenCV's cv.imdecode.
    
    Parameters:
    - url (str): URL of the image.
    - **kwargs: Additional keyword arguments for cv.imdecode (e.g., flags=cv.IMREAD_GRAYSCALE).
    
    Returns:
    - image: Loaded image as a NumPy array.
    """
    
    ### START CODE HERE ###
    arr = cv.imdecode(url, dtype=np.uint8)
    image = cv.imdecode(arr, -1)

    cv.imshow('Elephant walking', image)
    if cv.waitKey(): quit()
    ### END CODE HERE ###
    
    return image

parser = argparse.ArgumentParser(description="Load an image from a URL.")
parser.add_argument("--url", type=str, required=True, help="URL of the image to load.")
args = parser.parse_args()
load_image_from_url(args.url)