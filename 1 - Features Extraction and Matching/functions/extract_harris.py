import numpy as np
import cv2

from scipy import signal #for the scipy.signal.convolve2d function
from scipy import ndimage #for the scipy.ndimage.maximum_filter

# Harris corner detector
def extract_harris(img, image_name, sigma = 1.0, k = 0.05, thresh = 1e-5):
    '''
    Inputs:
    - img:      (h, w) gray-scaled image
    - sigma:    smoothing Gaussian sigma. suggested values: 0.5, 1.0, 2.0
    - k:        Harris response function constant. suggest interval: (0.04 - 0.06)
    - thresh:   scalar value to threshold corner strength. suggested interval: (1e-6 - 1e-4)
    Returns:
    - corners:  (q, 2) numpy array storing the keypoint positions [x, y]
    - C:     (h, w) numpy array storing the corner strength
    '''
    # Convert to float
    img = img.astype(float) / 255.0

    # 1. Compute image gradients in x and y direction
    dx = (1/2) * np.array([1, 0, -1]).reshape(1,3)
    dy = (1/2) * np.array([1, 0, -1]).reshape(3,1)
    Ix = signal.convolve2d(img, dx, mode='same', boundary='symm')
    Iy = signal.convolve2d(img, dy, mode='same', boundary='symm')
    
    # 2. Blur the computed gradients
    Ix_squared = Ix * Ix
    Iy_squared = Iy * Iy
    Ixy = Ix * Iy

    Ix_squared_smoothed = cv2.GaussianBlur(Ix_squared, (0, 0), sigma, borderType=cv2.BORDER_REPLICATE)
    Iy_squared_smoothed = cv2.GaussianBlur(Iy_squared, (0, 0), sigma, borderType=cv2.BORDER_REPLICATE)
    Ixy_smoothed = cv2.GaussianBlur(Ixy, (0, 0), sigma, borderType=cv2.BORDER_REPLICATE)

    # 3. Compute elements of the local auto-correlation matrix "M"
    M_determinant = Ix_squared_smoothed * Iy_squared_smoothed - pow(Ixy_smoothed, 2)
    M_trace = Ix_squared_smoothed + Iy_squared_smoothed

    # 4. Compute Harris response function C
    C = M_determinant - k * pow(M_trace, 2)

    # 5. Detection with threshold and non-maximum suppression
    C_thresholded = C > thresh
    local_maxima = ndimage.maximum_filter(C, size=3) == C
    corners_mask = C_thresholded & local_maxima
    y_coords, x_coords = np.where(corners_mask)
    corners = np.stack((x_coords, y_coords), axis=-1)
    
    # Visualize and save intermediate images
    '''
    images = {
        'Ix': Ix,
        'Iy': Iy,
        'Ixy': Ixy,
        'Ix_squared_smoothed': Ix_squared_smoothed,
        'Iy_squared_smoothed': Iy_squared_smoothed,
        'Ixy_smoothed': Ixy_smoothed,
        'C': C,
        'C_thresholded': C_thresholded.astype(float),
        'local_maxima': local_maxima.astype(float),
        'corner_mask': corners_mask.astype(float)
    }

    for key, value in images.items():
        normalized_img = cv2.normalize(value, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
        cv2.imwrite(f"{image_name}_{key}.png", normalized_img)
    '''
    
    return corners, C

