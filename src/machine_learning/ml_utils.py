import base64

import cv2
import numpy as np
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


def opencv_img_from_base64(im_base64):
    im_bytes = base64.b64decode(im_base64)
    im_arr = np.frombuffer(im_bytes, dtype=np.uint8)
    img = cv2.imdecode(im_arr, flags=cv2.IMREAD_COLOR)
    return img


def opencv_img_to_base64(img):
    _, im_arr = cv2.imencode('.jpg', img)
    im_bytes = im_arr.tobytes()
    im_base64 = base64.b64encode(im_bytes)
    return im_base64


def calculate_metrics_from_3x3_confusion_matrix(cm):
    tp = cm[0][0] + cm[1][1] + cm[2][2]
    tn = (cm[1][1] + cm[1][2] + cm[2][1] + cm[2][2]) + (cm[0][0] + cm[0][2] + cm[2][0] + cm[2][2]) + (
                cm[0][0] + cm[0][1] + cm[1][0] + cm[1][1])
    fp = (cm[1][0] + cm[2][0]) + (cm[0][1] + cm[2][1]) + (cm[0][2] + cm[1][2])
    fn = (cm[0][1] + cm[0][2]) + (cm[1][0] + cm[1][2]) + (cm[2][0] + cm[2][1])

    return int(tp), int(tn), int(fp), int(fn)
