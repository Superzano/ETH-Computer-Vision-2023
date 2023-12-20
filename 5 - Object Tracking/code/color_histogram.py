import cv2
import numpy as np

def color_histogram(xmin, ymin, xmax, ymax, frame, hist_bin):
    xmin = round(max(1, xmin))
    xmax = round(min(xmax, frame.shape[1]))
    ymin = round(max(1, ymin))
    ymax = round(min(ymax, frame.shape[0]))

    bounding_box_img = frame[ymin:ymax, xmin:xmax]

    hist_R = cv2.calcHist([bounding_box_img], [0], None, [hist_bin], [0, 256])
    hist_G = cv2.calcHist([bounding_box_img], [1], None, [hist_bin], [0, 256])
    hist_B = cv2.calcHist([bounding_box_img], [2], None, [hist_bin], [0, 256])

    hist = np.concatenate((hist_R, hist_G, hist_B)).flatten()
    hist = hist / sum(hist)

    return hist