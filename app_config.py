import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.svm import SVC


def preprocess_image(image_path):
    # Load image and convert to grayscale
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply median filter
    gray = cv2.medianBlur(gray, 3)

    # Enhance image
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Apply thresholding
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return binary


def load_data(data_dir):
    X = []
    y = []
    for subdir in os.listdir(data_dir):
        if subdir.startswith("stage"):
            label = int(subdir.split("_")[-1])
            for filename in os.listdir(os.path.join(data_dir, subdir)):
                image_path = os.path.join(data_dir, subdir, filename)
                image = preprocess_image(image_path)
                X.append(image.flatten())
                y.append(label)
    X = np.array(X)
    y = np.array(y)
    return X, y


# Load data
data_dir = "path/to/preprocessed/images"
X, y = load_data(data_dir)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train SVM
svm = SVC(kernel='linear', C=1, gamma='scale')
svm.fit(X_train, y_train)

# Test SVM
y_pred = svm.predict(X_test)
print(classification_report(y_test, y_pred))
