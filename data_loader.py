import os
from PIL import Image
import numpy as np

# Path to your dataset
data_dir = 'C:/Users/timmy/Desktop/Thyroid Ultrasound Metastasis Prediction/dataset'


# Function to load images
def load_images(data_folder):
    images_data = []
    labels_data = []
    for label in os.listdir(data_folder):
        label_dir = os.path.join(data_folder, label)
        if os.path.isdir(label_dir):
            for image_file in os.listdir(label_dir):
                image_path = os.path.join(label_dir, image_file)
                try:
                    image = Image.open(image_path)
                    image = image.resize((128, 128))  # Resize images to 128x128
                    image = np.array(image)  # Convert image to numpy array
                    if image.ndim == 2:  # Handle grayscale images
                        image = np.stack((image,) * 3, axis=-1)
                    images_data.append(image)
                    labels_data.append(label)
                except Exception as e:
                    print(f"Error reading {image_path}: {e}")
        else:
            print(f"Warning: {label_dir} is not a directory")
    return np.array(images_data), np.array(labels_data)


# Load images
images, labels = load_images(data_dir)

# Check the shape of loaded data
print(f"Images shape: {images.shape}")
print(f"Labels shape: {labels.shape}")

# Convert labels to binary (0 for non-metastatic, 1 for metastatic)
labels = np.where(labels == 'metastatic', 1, 0)
