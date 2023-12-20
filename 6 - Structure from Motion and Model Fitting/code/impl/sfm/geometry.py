import numpy as np

from impl.dlt import BuildProjectionConstraintMatrix
from impl.util import MakeHomogeneous, HNormalize
from impl.sfm.corrs import GetPairMatches
#from impl.opt import ImageResiduals, OptimizeProjectionMatrix

# # Debug
# import matplotlib.pyplot as plt
# from impl.vis import Plot3DPoints, PlotCamera, PlotProjectedPoints


def EstimateEssentialMatrix(K, im1, im2, matches):
  '''
  Normalize coordinates to the normalized image plane using the camera intrinsic matrix K
  This is done by making the keypoints homogeneous (adding a 1 as the third coordinate) 
  and then multiplying with the inverse of K
  '''
  normalized_kps1 = MakeHomogeneous(im1.kps, ax=1) @ np.linalg.inv(K).T
  normalized_kps2 = MakeHomogeneous(im2.kps, ax=1) @ np.linalg.inv(K).T

  # Assemble constraint matrix as equation 2.1
  constraint_matrix = np.zeros((matches.shape[0], 9))
  for i in range(matches.shape[0]):
    '''
    For each match, create a constraint from the normalized keypoints
    The constraint is based on the outer product of the corresponding points (x1 and x2)
    '''
    x1 = normalized_kps2[matches[i, 1], :]
    x2 = normalized_kps1[matches[i, 0], :]
    constraint_matrix[i] = np.kron(x1, x2).reshape(-1)
  
  # Solve for the nullspace of the constraint matrix
  _, _, vh = np.linalg.svd(constraint_matrix)
  vectorized_E_hat = vh[-1,:]

  '''
  Reshape the vectorized E_hat back into a 3x3 matrix
  '''
  E_hat = vectorized_E_hat.reshape(3, 3)

  '''
  Enforce the internal constraints of the essential matrix (E)
  The first two singular values are set to 1, and the third to 0 to satisfy the constraints of E
  '''
  u, _, vh = np.linalg.svd(E_hat)
  u, _, vh = np.linalg.svd(E_hat)
  E = u @ np.diag([1, 1, 0]) @ vh

  # This is just a quick test that should tell you if your estimated matrix is not correct
  # It might fail if you estimated E in the other direction (i.e. kp2' * E * kp1)
  # You can adapt it to your assumptions.
  for i in range(matches.shape[0]):
    kp1 = normalized_kps1[matches[i,0],:]
    kp2 = normalized_kps2[matches[i,1],:]

    assert(abs(kp2.transpose() @ E @ kp1) < 0.01)
  
  return E


def DecomposeEssentialMatrix(E):
  u, s, vh = np.linalg.svd(E)

  # Determine the translation up to sign
  t_hat = u[:,-1]

  W = np.array([
    [0, -1, 0],
    [1, 0, 0],
    [0, 0, 1]
  ])

  # Compute the two possible rotations
  R1 = u @ W @ vh
  R2 = u @ W.transpose() @ vh

  # Make sure the orthogonal matrices are proper rotations (Determinant should be 1)
  if np.linalg.det(R1) < 0:
    R1 *= -1

  if np.linalg.det(R2) < 0:
    R2 *= -1

  # Assemble the four possible solutions
  sols = [
    (R1, t_hat),
    (R2, t_hat),
    (R1, -t_hat),
    (R2, -t_hat)
  ]

  return sols


def TriangulatePoints(K, im1, im2, matches):

  R1, t1 = im1.Pose()
  R2, t2 = im2.Pose()
  P1 = K @ np.append(R1, np.expand_dims(t1, 1), 1)
  P2 = K @ np.append(R2, np.expand_dims(t2, 1), 1)

  # Ignore matches that already have a triangulated point
  new_matches = np.zeros((0, 2), dtype=int)

  num_matches = matches.shape[0]
  for i in range(num_matches):
    p3d_idx1 = im1.GetPoint3DIdx(matches[i, 0])
    p3d_idx2 = im2.GetPoint3DIdx(matches[i, 1])
    if p3d_idx1 == -1 and p3d_idx2 == -1:
      new_matches = np.append(new_matches, matches[[i]], 0)


  num_new_matches = new_matches.shape[0]

  points3D = np.zeros((num_new_matches, 3))

  for i in range(num_new_matches):
    kp1 = im1.kps[new_matches[i, 0], :]
    kp2 = im2.kps[new_matches[i, 1], :]

    # H & Z Sec. 12.2
    A = np.array([
      kp1[0] * P1[2] - P1[0],
      kp1[1] * P1[2] - P1[1],
      kp2[0] * P2[2] - P2[0],
      kp2[1] * P2[2] - P2[1]
    ])

    _, _, vh = np.linalg.svd(A)
    homogeneous_point = vh[-1]
    points3D[i] = homogeneous_point[:-1] / homogeneous_point[-1]


  # We need to keep track of the correspondences between image points and 3D points
  im1_corrs = new_matches[:,0]
  im2_corrs = new_matches[:,1]

  '''
  Transform the triangulated points into the coordinate space of the first camera
  MakeHomogeneous function converts the 3D points to homogeneous coordinates (adding a 1 as the fourth coordinate)
  These points are then multiplied with the transpose of the first camera's projection matrix (P1)
  '''
  cam1_space = MakeHomogeneous(points3D, 1) @ P1.T

  '''
  Filter points behind the first camera
  We only keep those points whose Z-coordinate (depth) in the camera coordinate space is positive
  Corresponding rows in im1_corrs and im2_corrs are also filtered to maintain consistency
  '''
  im1_corrs = im1_corrs[cam1_space[:, -1] > 0]
  im2_corrs = im2_corrs[cam1_space[:, -1] > 0]
  points3D = points3D[cam1_space[:, -1] > 0]

  '''
  Transform the filtered points into the coordinate space of the second camera
  This is the same as before but uses the second camera's projection matrix (P2)
  '''
  cam2_space = MakeHomogeneous(points3D, 1) @ P2.T

  '''
  Filter points behind the second camera
  Again we keep only those points that are in front of the second camera
  The correspondences in im1_corrs and im2_corrs are updated accordingly to match the filtered 3D points  
  '''
  im1_corrs = im1_corrs[cam2_space[:, -1] > 0]
  im2_corrs = im2_corrs[cam2_space[:, -1] > 0]
  points3D = points3D[cam2_space[:, -1] > 0]

  return points3D, im1_corrs, im2_corrs


def EstimateImagePose(points2D, points3D, K):  
  # We use points in the normalized image plane.
  # This removes the 'K' factor from the projection matrix.
  # We don't normalize the 3D points here to keep the code simpler.
  '''
  Normalize the 2D points by first converting them to homogeneous coordinates, then multiplying 
  with the transpose of the inverse of K.
  This operation transforms the 2D points from the image plane to the normalized image plane,
  removing the influence of the camera's intrinsic parameters
  '''
  normalized_points2D = MakeHomogeneous(points2D, 1) @ np.linalg.inv(K).T
  
  constraint_matrix = BuildProjectionConstraintMatrix(normalized_points2D, points3D)

  # We don't use optimization here since we would need to make sure to only optimize on the se(3) manifold
  # (the manifold of proper 3D poses). This is a bit too complicated right now.
  # Just DLT should give good enough results for this dataset.

  # Solve for the nullspace
  _, _, vh = np.linalg.svd(constraint_matrix)
  P_vec = vh[-1,:]
  P = np.reshape(P_vec, (3, 4), order='C')

  # Make sure we have a proper rotation
  u, s, vh = np.linalg.svd(P[:,:3])
  R = u @ vh

  if np.linalg.det(R) < 0:
    R *= -1

  _, _, vh = np.linalg.svd(P)
  C = np.copy(vh[-1,:])

  t = -R @ (C[:3] / C[3])

  return R, t


def TriangulateImage(K, image_name, images, registered_images, matches):
  # Loop over all registered images and triangulate new points with the new image.
  # Make sure to keep track of all new 2D-3D correspondences, also for the registered images

  image = images[image_name]
  points3D = np.zeros((0,3))
  # You can save the correspondences for each image in a dict and refer to the `local` new point indices here.
  # Afterwards you just add the index offset before adding the correspondences to the images.
  corrs = {}
  
  '''
  Initialize correspondence information for the new image
  Initialize end index of the last added points in points3D
  '''
  corrs[image_name] = {'points2D': []}
  prev_end = 0
  
  '''
  Loop through each registered image to triangulate points with the new image
  '''
  for reg_img_name in registered_images:
    reg_img = images[reg_img_name]
    
    '''
    Check the order of image names to ensure the match key is in the correct order
    '''
    if reg_img_name < image_name:
      '''
      Triangulate points between the registered image and the new image
      '''
      reg_points3D, reg_img_corrs, img_corrs = TriangulatePoints(K, reg_img, image, matches[(reg_img_name, image_name)])
    else:
      '''
      Triangulate points, ensuring the new image is the first parameter if its name comes after the registered image's name
      '''
      reg_points3D, img_corrs, reg_img_corrs = TriangulatePoints(K, image, reg_img, matches[(image_name, reg_img_name)])
    
    '''
    Append the newly triangulated 3D points to the points3D array
    '''
    points3D = np.append(points3D, reg_points3D, 0)
    
    '''
    Update the correspondences dictionary for the registered image
    Store which 2D keypoints in the registered image correspond to the newly added 3D points
    '''
    corrs[reg_img_name] = {}
    corrs[reg_img_name]['points2D'] = reg_img_corrs
    corrs[reg_img_name]['interval'] = prev_end, points3D.shape[0]
    
    '''
    Update the correspondences for the new image
    '''
    corrs[image_name]['points2D'].extend(img_corrs)
    
    '''
    Update prev_end to mark the new end of the points3D array
    '''
    prev_end = points3D.shape[0]

  '''
  Store the overall interval of the 3D points corresponding to the new image
  '''
  corrs[image_name]['interval'] = 0, points3D.shape[0]
  
  return points3D, corrs

