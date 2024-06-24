import cv2
import numpy as np
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import os
import matplotlib.pyplot as plt

# Fungsi untuk segmentasi gambar
def segment_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return binary

# Fungsi untuk ekstraksi fitur
def extract_features(image):
    features, hog_image = hog(image, pixels_per_cell=(16, 16),
                              cells_per_block=(2, 2), visualize=True)
    return features

# Fungsi untuk memuat dan memproses gambar
def load_and_process_images(image_paths):
    features = []
    for path in image_paths:
        image = cv2.imread(path)
        segmented_image = segment_image(image)
        feature = extract_features(segmented_image)
        features.append(feature)
    return np.array(features)

# Memuat label dari file teks
dataset_path = 'synthetic_dataset'
labels_file = os.path.join(dataset_path, 'labels.txt')

image_paths = []
labels = []
with open(labels_file, 'r') as f:
    for line in f:
        path, label = line.strip().split(',')
        image_paths.append(path)
        labels.append(int(label))

# Memuat dan memproses gambar
features = load_and_process_images(image_paths)
labels = np.array(labels)

# Bagi data menjadi set pelatihan dan set pengujian
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Klasifikasi menggunakan Random Forest
clf_rf = RandomForestClassifier(n_estimators=100)
clf_rf.fit(X_train, y_train)
y_pred_rf = clf_rf.predict(X_test)
print("Random Forest Classification Report")
print(classification_report(y_test, y_pred_rf))

# Klasifikasi menggunakan SVM
clf_svm = SVC(kernel='linear')
clf_svm.fit(X_train, y_train)
y_pred_svm = clf_svm.predict(X_test)
print("SVM Classification Report")
print(classification_report(y_test, y_pred_svm))

# Visualisasi hasil segmentasi dan ekstraksi fitur
for path in image_paths[:5]:  # Visualisasi 5 gambar pertama
    image = cv2.imread(path)
    segmented_image = segment_image(image)
    _, hog_image = hog(segmented_image, pixels_per_cell=(16, 16),
                       cells_per_block=(2, 2), visualize=True)
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title('Segmented Image')
    plt.imshow(segmented_image, cmap='gray')
    plt.subplot(1, 2, 2)
    plt.title('HOG Features')
    plt.imshow(hog_image, cmap='gray')
    plt.show()
