import cv2
from skimage import feature


def fd_hu_moments(image):
    return cv2.HuMoments(cv2.moments(image)).flatten()


def quantify_image(image):
    # For Speed and pressure of signature image
    # compute the histogram of oriented gradients feature vector for
    # the input image
    features = feature.hog(image, orientations=9,
                           pixels_per_cell=(10, 10), cells_per_block=(2, 2),
                           transform_sqrt=True, block_norm="L1")

    # return the feature vector
    return features
