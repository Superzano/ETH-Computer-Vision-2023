import numpy as np
import cv2
import glob
import os
from sklearn.cluster import KMeans
from tqdm import tqdm

# My imports
import utils as uts
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import pickle

def findnn(D1, D2):
    """
    :param D1: NxD matrix containing N feature vectors of dim. D
    :param D2: MxD matrix containing M feature vectors of dim. D
    :return:
        Idx: N-dim. vector containing for each feature vector in D1 the index of the closest feature vector in D2.
        Dist: N-dim. vector containing for each feature vector in D1 the distance to the closest feature vector in D2
    """
    N = D1.shape[0]
    M = D2.shape[0]  # [k]

    # Find for each feature vector in D1 the nearest neighbor in D2
    Idx, Dist = [], []
    for i in range(N):
        minidx = 0
        mindist = np.linalg.norm(D1[i, :] - D2[0, :])
        for j in range(1, M):
            d = np.linalg.norm(D1[i, :] - D2[j, :])

            if d < mindist:
                mindist = d
                minidx = j
        Idx.append(minidx)
        Dist.append(mindist)
    return Idx, Dist


def grid_points(img, nPointsX, nPointsY, border):
    """
    :param img: input gray img, numpy array, [h, w]
    :param nPointsX: number of grids in x dimension
    :param nPointsY: number of grids in y dimension
    :param border: leave border pixels in each image dimension
    :return: vPoints: 2D grid point coordinates, numpy array, [nPointsX*nPointsY, 2]
    """
    
    height, width = img.shape[:2]
    x_points = np.linspace(border + 1, width - border, nPointsX, dtype=int)
    y_points = np.linspace(border + 1, height - border, nPointsY, dtype=int)
    x_points_grid, y_points_grid = np.meshgrid(x_points, y_points)
    vPoints = np.vstack((x_points_grid.flatten(), y_points_grid.flatten())).T

    return vPoints



def descriptors_hog(img, vPoints, cellWidth, cellHeight):
    nBins = 8
    w = cellWidth
    h = cellHeight

    grad_x = cv2.Sobel(img, cv2.CV_16S, dx=1, dy=0, ksize=1)
    grad_y = cv2.Sobel(img, cv2.CV_16S, dx=0, dy=1, ksize=1)
    
    grad_x = np.float32(grad_x)
    grad_y = np.float32(grad_y)
    
    #magnitude = cv2.magnitude(grad_x, grad_y)
    orientation = cv2.phase(grad_x, grad_y, angleInDegrees=False)

    descriptors = []
    for i in range(len(vPoints)):
        center_x = round(vPoints[i, 0])
        center_y = round(vPoints[i, 1])

        desc = []
        for cell_y in range(-2, 2):
            for cell_x in range(-2, 2):
                start_y = center_y + (cell_y) * h
                end_y = center_y + (cell_y + 1) * h

                start_x = center_x + (cell_x) * w
                end_x = center_x + (cell_x + 1) * w
                
                cell_orientation = orientation[start_y:end_y, start_x:end_x]
                #cell_magnitude = magnitude[start_y:end_y, start_x:end_x]
                cell_orientation = cell_orientation.flatten()
                #cell_magnitude = cell_magnitude.flatten()
                hist, _ = np.histogram(cell_orientation, bins=nBins, range=(-np.pi, np.pi))
                #hist, _ = np.histogram(cell_orientation, bins=nBins, range=(-np.pi, np.pi), weights=cell_magnitude)
                desc.extend(hist.astype(np.float32))

        descriptors.append(desc)

    descriptors = np.asarray(descriptors)
    
    return descriptors




def create_codebook(nameDirPos, nameDirNeg, k, numiter):
    """
    :param nameDirPos: dir to positive training images
    :param nameDirNeg: dir to negative training images
    :param k: number of kmeans cluster centers
    :param numiter: maximum iteration numbers for kmeans clustering
    :return: vCenters: center of kmeans clusters, numpy array, [k, 128]
    """
    vImgNames = sorted(glob.glob(os.path.join(nameDirPos, '*.png')))
    vImgNames = vImgNames + sorted(glob.glob(os.path.join(nameDirNeg, '*.png')))

    nImgs = len(vImgNames)

    cellWidth = 4
    cellHeight = 4
    nPointsX = 10
    nPointsY = 10
    border = 8

    vFeatures = []
    for i in tqdm(range(nImgs)):
        #print('\nprocessing image {} ...'.format(i+1))
        img = cv2.imread(vImgNames[i])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        vPoints = grid_points(img, nPointsX, nPointsY, border)
        # Plot the grid on top of input image for visualization
        #uts.plot_grid(img, vPoints)
        descriptors = descriptors_hog(img, vPoints, cellWidth, cellHeight)
        
        vFeatures.extend(descriptors)
    vFeatures = np.asarray(vFeatures)
    vFeatures = vFeatures.reshape(-1, vFeatures.shape[-1])
    print('number of extracted features: ', len(vFeatures))

    print('clustering ...')
    kmeans_res = KMeans(n_clusters=k, max_iter=numiter, n_init='auto', random_state=42).fit(vFeatures)
    vCenters = kmeans_res.cluster_centers_
    return vCenters


def bow_histogram(vFeatures, vCenters):
    """
    :param vFeatures: MxD matrix containing M feature vectors of dim. D
    :param vCenters: NxD matrix containing N cluster centers of dim. D
    :return: histo: N-dim. numpy vector containing the resulting BoW activation histogram.
    """
    
    N = vCenters.shape[0]
    histo = np.zeros(N, dtype=int)
    idx, _ = findnn(vFeatures, vCenters)
    for i in idx:
        histo[i] += 1
        
    return histo





def create_bow_histograms(nameDir, vCenters):
    """
    :param nameDir: dir of input images
    :param vCenters: kmeans cluster centers, [k, 128] (k is the number of cluster centers)
    :return: vBoW: matrix, [n_imgs, k]
    """
    vImgNames = sorted(glob.glob(os.path.join(nameDir, '*.png')))
    nImgs = len(vImgNames)

    cellWidth = 4
    cellHeight = 4
    nPointsX = 10
    nPointsY = 10
    border = 8

    vBoW = []
    for i in tqdm(range(nImgs)):
        # print('processing image {} ...'.format(i + 1))
        img = cv2.imread(vImgNames[i])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        vPoints = grid_points(img, nPointsX, nPointsY, border)
        descriptors = descriptors_hog(img, vPoints, cellWidth, cellHeight)
        histogram = bow_histogram(descriptors, vCenters)
        vBoW.append(histogram)
        
    vBoW = np.asarray(vBoW)

    return vBoW



def bow_recognition_nearest(histogram,vBoWPos,vBoWNeg):
    """
    :param histogram: bag-of-words histogram of a test image, [1, k]
    :param vBoWPos: bag-of-words histograms of positive training images, [n_imgs, k]
    :param vBoWNeg: bag-of-words histograms of negative training images, [n_imgs, k]
    :return: sLabel: predicted result of the test image, 0(without car)/1(with car)
    """

    _ , DistPos = findnn(histogram, vBoWPos)
    _ , DistNeg = findnn(histogram, vBoWNeg)

    if DistPos[0] < DistNeg[0]:
        sLabel = 1
    else:
        sLabel = 0
        
    return sLabel





if __name__ == '__main__':
    nameDirPos_train = 'data/data_bow/cars-training-pos'
    nameDirNeg_train = 'data/data_bow/cars-training-neg'
    nameDirPos_test = 'data/data_bow/cars-testing-pos'
    nameDirNeg_test = 'data/data_bow/cars-testing-neg'

    # Uncomment the following part to perform hparam tuning over k (KMeans)
    '''
    start, end, step = 10, 250, 10
    k_values = list(range(start, end + 1, step))
    acc_pos_list = []
    acc_neg_list = []
    '''
    k_values = [190] #I decided to keep k=190 after hparam tuning (see report)
    numiter = 300

    for k in k_values:
        #print(f'creating codebook for k={k} ...')
        print('creating codebook ...')
        vCenters = create_codebook(nameDirPos_train, nameDirNeg_train, k, numiter)
        
        print('creating bow histograms (pos) ...')
        vBoWPos = create_bow_histograms(nameDirPos_train, vCenters)
        print('creating bow histograms (neg) ...')
        vBoWNeg = create_bow_histograms(nameDirNeg_train, vCenters)
        
        # test pos samples
        print('creating bow histograms for test set (pos) ...')
        vBoWPos_test = create_bow_histograms(nameDirPos_test, vCenters)  # [n_imgs, k]
        result_pos = 0
        print('testing pos samples ...')
        
            
        for i in range(vBoWPos_test.shape[0]):
            cur_label = bow_recognition_nearest(vBoWPos_test[i:(i+1)], vBoWPos, vBoWNeg)
            result_pos = result_pos + cur_label
        acc_pos = result_pos / vBoWPos_test.shape[0]
        #acc_pos_list.append(acc_pos)
        print('test pos sample accuracy:', acc_pos)
        

        # test neg samples
        print('creating bow histograms for test set (neg) ...')
        vBoWNeg_test = create_bow_histograms(nameDirNeg_test, vCenters)  # [n_imgs, k]
        result_neg = 0
        print('testing neg samples ...')
        
        
        for i in range(vBoWNeg_test.shape[0]):
            cur_label = bow_recognition_nearest(vBoWNeg_test[i:(i + 1)], vBoWPos, vBoWNeg)
            result_neg = result_neg + cur_label
        acc_neg = 1 - result_neg / vBoWNeg_test.shape[0]
        #acc_neg_list.append(acc_neg)
        print('test neg sample accuracy:', acc_neg)
        
        #print(f'k={k} | Positive sample accuracy: {acc_pos} | Negative sample accuracy: {acc_neg}')
    
    # Code for hparameter tuning visualization
    '''
    # Save the accuracies lists to a file using pickle
    with open('accuracies.pkl', 'wb') as f:
        pickle.dump((k_values, acc_pos_list, acc_neg_list), f)

    #print('Accuracies saved to accuracies.pkl')

    # Load the accuracies from the file
    with open('accuracies.pkl', 'rb') as f:
        k_values, pos_acc_list, neg_acc_list = pickle.load(f)
    
    # Plot the accuracies
    
    plt.figure(figsize=(10, 5))
    plt.plot(k_values, pos_acc_list, marker='o', linestyle='-', color='b', label='Positive accuracy')
    plt.plot(k_values, neg_acc_list, marker='s', linestyle='--', color='r', label='Negative accuracy')
    plt.xlabel('k')
    plt.ylabel('Accuracy')
    plt.title('Positive and Negative Sample Accuracies for Different k Values')
    plt.legend()
    plt.grid(True)
    plt.xticks(k_values[::2])
    
    # Save the figure
    plt.savefig('accuracies_plot.png', dpi=300)

    plt.show()
    '''
    