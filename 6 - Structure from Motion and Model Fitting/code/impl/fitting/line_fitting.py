import numpy as np
from matplotlib import pyplot as plt
import random

np.random.seed(0)
random.seed(0)

def least_square(x,y):
	# return the least-squares solution
	# you can use np.linalg.lstsq
	'''
	Stack x and a column of ones to create a two-column matrix where the first column is x and the second is the bias term.
    This returrns the coefficients that minimize the square of the difference
    between the observed outputs in y and the outputs predicted by our linear model
 	'''

	k, b = np.linalg.lstsq(np.stack([x, np.ones_like(x)], axis = 1), y)[0]
	return k, b

def num_inlier(x,y,k,b,n_samples,thres_dist):
	# compute the number of inliers and a mask that denotes the indices of inliers
	'''
    Compute the number of inliers by first predicting the y values using the model's k and b (y_hat), 
    then comparing the predicted y values against the actual y values. 
    Points with a squared difference less than thres_dist are considered inliers
    '''
	num = 0
	mask = np.zeros(x.shape, dtype=bool)
 
	y_hat = k * x + b
	squared_loss = np.square(y - y_hat)
	mask = squared_loss < thres_dist
	num = np.count_nonzero(mask)

	return num, mask

def ransac(x,y,iter,n_samples,thres_dist,num_subset):
    # ransac
	k_ransac = None
	b_ransac = None
	inlier_mask = None
	best_inliers = 0
 
	idx_range = np.arange(len(x))
	for _ in range(iter):
		'''
		Randomly select a subset of points to calculate a potential model
		'''
		idx_sample = np.random.choice(idx_range, num_subset, replace=False)
		cur_k, cur_b = least_square(x[idx_sample], y[idx_sample])

		'''
		Determine the number of inliers and the inlier mask
  		'''
		cur_inlier, cur_mask = num_inlier(x, y, cur_k, cur_b, n_samples, thres_dist)

		'''
  		If the model has the most inliers so far then update the best model to these parameters
    	'''
		if cur_inlier > best_inliers:
			k_ransac, b_ransac = cur_k, cur_b
			best_inliers = cur_inlier
			inlier_mask = cur_mask

	return k_ransac, b_ransac, inlier_mask

def main():
	iter = 300
	thres_dist = 1
	n_samples = 500
	n_outliers = 50
	k_gt = 1
	b_gt = 10
	num_subset = 5
	x_gt = np.linspace(-10,10,n_samples)
	print(x_gt.shape)
	y_gt = k_gt*x_gt+b_gt
	# add noise
	x_noisy = x_gt+np.random.random(x_gt.shape)-0.5
	y_noisy = y_gt+np.random.random(y_gt.shape)-0.5
	# add outlier
	x_noisy[:n_outliers] = 8 + 10 * (np.random.random(n_outliers)-0.5)
	y_noisy[:n_outliers] = 1 + 2 * (np.random.random(n_outliers)-0.5)

	# least square
	k_ls, b_ls = least_square(x_noisy, y_noisy)

	# ransac
	k_ransac, b_ransac, inlier_mask = ransac(x_noisy, y_noisy, iter, n_samples, thres_dist, num_subset)
	outlier_mask = np.logical_not(inlier_mask)

	print("Estimated coefficients (true, linear regression, RANSAC):")
	print(k_gt, b_gt, k_ls, b_ls, k_ransac, b_ransac)

	line_x = np.arange(x_noisy.min(), x_noisy.max())
	line_y_ls = k_ls*line_x+b_ls
	line_y_ransac = k_ransac*line_x+b_ransac

	plt.scatter(
	    x_noisy[inlier_mask], y_noisy[inlier_mask], color="yellowgreen", marker=".", label="Inliers"
	)
	plt.scatter(
	    x_noisy[outlier_mask], y_noisy[outlier_mask], color="gold", marker=".", label="Outliers"
	)
	plt.plot(line_x, line_y_ls, color="navy", linewidth=2, label="Linear regressor")
	plt.plot(
	    line_x,
	    line_y_ransac,
	    color="cornflowerblue",
	    linewidth=2,
	    label="RANSAC regressor",
	)
	plt.legend()
	plt.xlabel("Input")
	plt.ylabel("Response")
	plt.show()

if __name__ == '__main__':
	main()