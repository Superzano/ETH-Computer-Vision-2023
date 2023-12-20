import os
import cv2
import pdb

from functions.extract_harris import extract_harris
from functions.extract_descriptors import filter_keypoints, extract_patches
from functions.match_descriptors import match_descriptors
from functions.vis_utils import plot_image_with_keypoints, plot_image_pair_with_matches
from itertools import product #for "hyperparameters tuning" (combinations of values suggested ranges)

# constants
HARRIS_SIGMA = 2.0
HARRIS_K = 0.06
HARRIS_THRESH = 1e-6
MATCHING_RATIO_TEST_THRESHOLD = 0.5

def main_detection():
    
    IMG_NAME1 = "images/blocks.jpg"
    IMG_NAME2 = "images/house.jpg"
    
    '''
    ### [OLD] Code for Hparams tuning ###
    # Hyperparameters range
    sigmas = [0.5, 1.0, 2.0]
    ks = [0.04, 0.045, 0.05, 0.055, 0.06]
    threshs = [1e-6, 1e-5, 1e-4]
        
    # Generate combinations of hyperparameters
    hyperparameter_combinations = product(sigmas, ks, threshs)
        
    # Output Directory
    output_dir_blocks = "harris_results/blocks"
    output_dir_house = "harris_results/house"
    os.makedirs(output_dir_blocks, exist_ok=True)
    os.makedirs(output_dir_house, exist_ok=True)
    
    for sigma, k, thresh in hyperparameter_combinations:
        # Harris corner detection
        corners1, C1 = extract_harris(img1, sigma, k, thresh)
        # Image title
        title = f"sigma={sigma}_k={k}_thresh={thresh}"
        # Save the image
        output_path = os.path.join(output_dir_blocks, f"{title}")
        plot_image_with_keypoints(output_path, img1, corners1)
        print(f"Saved image1: {output_path}")

        corners2, C2 = extract_harris(img2, sigma, k, thresh)
        # Image title
        title = f"sigma={sigma}_k={k}_thresh={thresh}"
        # Save the image
        output_path = os.path.join(output_dir_house, f"{title}")
        plot_image_with_keypoints(output_path, img2, corners2)
        print(f"Saved image2: {output_path}")
    '''

    # Harris corner detection
    img1 = cv2.imread(IMG_NAME1, cv2.IMREAD_GRAYSCALE)
    corners1, C1 = extract_harris(img1, "blocks", HARRIS_SIGMA, HARRIS_K, HARRIS_THRESH)
    plot_image_with_keypoints(os.path.basename(IMG_NAME1[:-4]) + "_harris.png", img1, corners1)

    img2 = cv2.imread(IMG_NAME2, cv2.IMREAD_GRAYSCALE)
    corners2, C2 = extract_harris(img2, "house", HARRIS_SIGMA, HARRIS_K, HARRIS_THRESH)
    plot_image_with_keypoints(os.path.basename(IMG_NAME2[:-4]) + "_harris.png", img2, corners2)

def main_matching():
    IMG_NAME1 = "images/I1.jpg"
    IMG_NAME2 = "images/I2.jpg"
    
    '''
    ### [OLD] Code for Hparams tuning ###
    # Hyperparameters range
    sigmas = [0.5, 1.0, 2.0]
    ks = [0.04, 0.05, 0.06]
    threshs = [1e-6, 1e-5, 1e-4]
    matching_ratio_threshs = [0.2, 0.5, 0.8]
        
    # Generate combinations of hyperparameters
    hyperparameter_combinations = product(sigmas, ks, threshs)

    # Output Directory
    output_dir_I1 = "harris_results/I1"
    output_dir_I2 = "harris_results/I2"
    os.makedirs(output_dir_I1, exist_ok=True)
    os.makedirs(output_dir_I2, exist_ok=True)
    
    for sigma, k, thresh in hyperparameter_combinations:
        # Harris corner detection
        corners1, C1 = extract_harris(img1, sigma, k, thresh)
        # Image title
        title = f"sigma={sigma}_k={k}_thresh={thresh}"
        # Save the image
        output_path = os.path.join(output_dir_I1, f"{title}")
        plot_image_with_keypoints(output_path, img1, corners1)
        print(f"Saved image1: {output_path}")
        
        corners2, C2 = extract_harris(img2, sigma, k, thresh)
        # Image title
        title = f"sigma={sigma}_k={k}_thresh={thresh}"
        # Save the image
        output_path = os.path.join(output_dir_I2, f"{title}")
        plot_image_with_keypoints(output_path, img2, corners2)
        print(f"Saved image2: {output_path}")
    '''
    
    # Harris corner detection
    img1 = cv2.imread(IMG_NAME1, cv2.IMREAD_GRAYSCALE)
    corners1, C1 = extract_harris(img1, "I1", HARRIS_SIGMA, HARRIS_K, HARRIS_THRESH)
    plot_image_with_keypoints(os.path.basename(IMG_NAME1[:-4]) + "_harris.png", img1, corners1)

    img2 = cv2.imread(IMG_NAME2, cv2.IMREAD_GRAYSCALE)
    corners2, C2 = extract_harris(img2, "I2", HARRIS_SIGMA, HARRIS_K, HARRIS_THRESH)
    plot_image_with_keypoints(os.path.basename(IMG_NAME2[:-4]) + "_harris.png", img2, corners2)

    # Extract descriptors
    corners1 = filter_keypoints(img1, corners1, patch_size=9)
    desc1 = extract_patches(img1, corners1, patch_size=9)
    corners2 = filter_keypoints(img2, corners2, patch_size=9)
    desc2 = extract_patches(img2, corners2, patch_size=9)
    # Matching
    matches_ow = match_descriptors(desc1, desc2, "one_way")
    plot_image_pair_with_matches("match_ow.png", img1, corners1, img2, corners2, matches_ow)
    matches_mutual = match_descriptors(desc1, desc2, "mutual")
    plot_image_pair_with_matches("match_mutual.png", img1, corners1, img2, corners2, matches_mutual)
    matches_ratio = match_descriptors(desc1, desc2, "ratio", ratio_thresh=MATCHING_RATIO_TEST_THRESHOLD)
    plot_image_pair_with_matches("match_ratio.png", img1, corners1, img2, corners2, matches_ratio)

def main():
    main_detection()
    pdb.set_trace() # Enter c to continue to matching, q to exit.
    main_matching()

if __name__ == "__main__":
    main()

