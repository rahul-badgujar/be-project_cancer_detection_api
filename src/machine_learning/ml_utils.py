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


def read_image_in_greyscale_as_np_array(img_path):
    img = cv2.imread(img_path)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray_img


def write_np_array_as_image(img_array, out_img_path):
    is_saved = cv2.imwrite(out_img_path, img_array)
    if is_saved:
        return out_img_path
    else:
        raise Exception("Failed to save output image")
