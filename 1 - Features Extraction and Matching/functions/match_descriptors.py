import numpy as np

def ssd(desc1, desc2):
    '''
    Sum of squared differences
    Inputs:
    - desc1:        - (q1, feature_dim) descriptor for the first image
    - desc2:        - (q2, feature_dim) descriptor for the first image
    Returns:
    - distances:    - (q1, q2) numpy array storing the squared distance
    '''
    assert desc1.shape[1] == desc2.shape[1]
    
    desc1 = desc1[:, np.newaxis, :]
    desc2 = desc2[np.newaxis, :, :]
    squared_diffs = pow((desc1 - desc2), 2)
    distances = np.sum(squared_diffs, axis=2)
    
    return distances

def match_descriptors(desc1, desc2, method = "one_way", ratio_thresh=0.5):
    '''
    Match descriptors
    Inputs:
    - desc1:        - (q1, feature_dim) descriptor for the first image
    - desc2:        - (q2, feature_dim) descriptor for the first image
    Returns:
    - matches:      - (m x 2) numpy array storing the indices of the matches
    '''
    assert desc1.shape[1] == desc2.shape[1]
    distances = ssd(desc1, desc2)
    q1, q2 = desc1.shape[0], desc2.shape[0]
    matches = None
    
    if method == "one_way":
        min_indices = np.argmin(distances, axis=1)
        matches = np.column_stack((np.arange(q1), min_indices))
        
    elif method == "mutual":
        min_indices_1 = np.min(distances, axis=1)
        min_indices_2 = np.min(distances, axis=0)
        mutual_mask = (min_indices_1[:, np.newaxis] == min_indices_2)
        matches = np.column_stack(np.where(mutual_mask))
        
    elif method == "ratio":
        second_min_distances = np.partition(distances, 2, axis=1)[:, 1]
        distance_ratio = distances[np.arange(q1), np.argmin(distances, axis=1)] / second_min_distances
        ratio_mask = (distance_ratio < ratio_thresh)
        matches = np.column_stack((np.where(ratio_mask)[0], np.argmin(distances, axis=1)[ratio_mask]))
        
    else:
       raise ValueError("Unknown matching method.")
   
    return matches

