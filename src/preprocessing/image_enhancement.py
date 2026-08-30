"""
OpenCV Radiographic Image Enhancement Module
Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) and bilateral filtering.
"""
import cv2
import numpy as np

def apply_clahe(image: np.ndarray, clip_limit: float = 2.0, tile_grid_size: tuple = (8, 8)) -> np.ndarray:
    """Enhance contrast of dental panoramic X-ray using CLAHE."""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced = clahe.apply(gray)

    # Convert back to 3-channel for CNN backbones
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)

def preprocess_xray_image(image_path: str, target_size: tuple = (224, 224)) -> np.ndarray:
    """Full preprocessing pipeline: load -> resize -> CLAHE -> normalize."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f'Unable to read image at {image_path}')

    resized = cv2.resize(img, target_size)
    enhanced = apply_clahe(resized)
    normalized = enhanced.astype(np.float32) / 255.0
    return normalized