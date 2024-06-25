# split_data.py
import numpy as np
from sklearn.model_selection import train_test_split
from data_loader import load_images

# Path to your dataset
data_dir = 'C:/Users/timmy/Desktop/Thyroid Ultrasound Metastasis Prediction/dataset'

# Load images and labels
images, labels = load_images(data_dir)

# Convert labels to binary (0 for non-metastatic, 1 for metastatic)
labels = np.where(labels == 'metastatic', 1, 0)

# Split the dataset into training, validation, and test sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)  # 0.25 x 0.8 = 0.2

# Save the data splits
np.savez_compressed('data_splits.npz', X_train=X_train, X_val=X_val, X_test=X_test, y_train=y_train, y_val=y_val, y_test=y_test)
