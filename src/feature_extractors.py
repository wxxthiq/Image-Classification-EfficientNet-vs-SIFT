import cv2
import numpy as np

def extract_sift_dog(image, contrast_threshold=0.04, edge_threshold=10):
    """Extracts keypoints and SIFT descriptors using DoG detector."""
    sift = cv2.SIFT_create(contrastThreshold=contrast_threshold, edgeThreshold=edge_threshold)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    return keypoints, descriptors

def extract_harris_laplace(image, block_size=3, k=0.04, sigma_values=[1.0, 1.5, 2.0], contrast_threshold=0.04, edge_threshold=10):
    """Extracts keypoints using Harris-Laplace and SIFT descriptors."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 1. Harris corner detection for initial keypoints
    harris_response = cv2.cornerHarris(gray, blockSize=block_size, ksize=3, k=k)
    coords = np.where(harris_response > 0.01 * harris_response.max())
    keypoints = [cv2.KeyPoint(float(x), float(y), 1.0) for y, x in zip(coords[0], coords[1])]

    if not keypoints:
        return [], None

    # 2. Scale selection with Laplacian of Gaussian (LoG)
    best_sigma = None
    best_response = -float('inf')
    for sigma in sigma_values:
        log = cv2.GaussianBlur(gray, (0, 0), sigma)
        log_response = np.array([log[int(kp.pt[1]), int(kp.pt[0])] for kp in keypoints if 0 <= int(kp.pt[1]) < gray.shape[0] and 0 <= int(kp.pt[0]) < gray.shape[1]])
        if log_response.size > 0:
            max_response = np.max(log_response)
            if max_response > best_response:
                best_response = max_response
                best_sigma = sigma
    
    if best_sigma is None:
        best_sigma = sigma_values[0]

    for kp in keypoints:
        kp.size = best_sigma * 2

    # 3. Compute SIFT descriptors
    sift = cv2.SIFT_create(contrastThreshold=contrast_threshold, edgeThreshold=edge_threshold)
    keypoints, descriptors = sift.compute(gray, keypoints)
    
    return keypoints, descriptors
