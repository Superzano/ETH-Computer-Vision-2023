import time
import numpy as np

# run `pip install scikit-image` to install skimage, if you haven't done so
from skimage import io, color
from skimage.transform import rescale

# Change here the bandwidth [1, 2.5, 3, 5, 7]
BANDWIDTH = 2.5
SOLUTION_METHOD = "centroids_limitation"    # "centroids_limitation" or "centroids_colors_mapping"

# Function to solve the bandwith=1 problem using the "centroids_colors_mapping" method
def find_nearest_color(centroid, colors):
    distances = np.linalg.norm(colors - centroid, axis=1)
    return np.argmin(distances)

def distance(x, X):
    return np.linalg.norm(X - x, axis=1)

def gaussian(dist, bandwidth):
    return  (np.exp(-0.5 * (dist ** 2) / (bandwidth ** 2))) / (bandwidth * np.sqrt(2 * np.pi))

def update_point(weight, X):
    return (np.sum(weight[:, np.newaxis] * X, axis=0)) / np.sum(weight)

def meanshift_step(X, bandwidth):
    new_X = np.zeros_like(X)
    for i, x in enumerate(X):                              
        distances = distance(x, X)
        weights = gaussian(distances, bandwidth)
        new_X[i] = update_point(weights, X)
    return new_X

def meanshift(X):
    for i in range(20):
        print("Iteration Number: ", i)
        X = meanshift_step(X, BANDWIDTH)
    return X

scale = 0.5    # downscale the image to run faster

# Load image and convert it to CIELAB space
image = rescale(io.imread('eth.jpg'), scale, channel_axis=-1)
image_lab = color.rgb2lab(image)
shape = image_lab.shape # record image shape
image_lab = image_lab.reshape([-1, 3])  # flatten the image

# Run your mean-shift algorithm
t = time.time()
X = meanshift(image_lab)
t = time.time() - t
print ('Elapsed time for mean-shift: {}'.format(t))

# Load label colors and draw labels as an image
colors = np.load('colors.npz')['colors']
colors[colors > 1.0] = 1
colors[colors < 0.0] = 0

centroids, labels = np.unique((X / 4).round(), return_inverse=True, axis=0)

print("Number of centroids: ", centroids.shape)
print("Number of labels: ", np.max(labels))

# Note: the following code is to make the bandwith=1 work according to the solutions discussed in the report.
if centroids.shape[0] > 24:
    if SOLUTION_METHOD == "centroids_limitation":
        print("Performing centroids_limitation solution ...")
        unique_centroids, counts = np.unique(labels, return_counts=True)
        sorted_centroids = unique_centroids[np.argsort(-counts)][:24]
        mapped_labels = np.array([np.argmin(np.abs(sorted_centroids - label)) for label in labels])
        
        result_image = colors[mapped_labels].reshape(shape)
        result_image = rescale(result_image, 1 / scale, order=0, channel_axis=-1)     # resize result image to original resolution
        result_image = (result_image * 255).astype(np.uint8)
        io.imsave('result_limited_centroids.png', result_image)
        
    elif SOLUTION_METHOD == "centroids_colors_mapping":
        print("Performing centroids_colors_mapping solution ...")
        mapped_labels = np.array([find_nearest_color(centroid, colors) for centroid in centroids])
        result_image = colors[mapped_labels[labels]].reshape(shape)
        result_image = rescale(result_image, 1 / scale, order=0, channel_axis=-1)     # resize result image to original resolution
        result_image = (result_image * 255).astype(np.uint8)
        io.imsave('result_mapped_colors.png', result_image)
        
    else:
        print("[ERROR] Solution method for bandwidth problem is not properly defined!")
else:
    result_image = colors[labels].reshape(shape)
    result_image = rescale(result_image, 1 / scale, order=0, channel_axis=-1)     # resize result image to original resolution
    result_image = (result_image * 255).astype(np.uint8)
    io.imsave('result.png', result_image)




