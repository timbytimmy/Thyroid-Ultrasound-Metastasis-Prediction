# Thyroid-Ultrasound-Metastasis-Prediction
perform "metastatic" and "non-metastatic" classification on thyroid oltrasound images using python

## Overview
The project uses a Random Forest Classifier to classify thyroid ultrasound images. The images are preprocessed, split into training, validation, and test sets, and then fed into the classifier. The model's performance is evaluated using accuracy and a confusion matrix.

## Dataset
The dataset consists of thyroid ultrasound images categorized into two classes: "metastatic" and "non-metastatic". The images are resized to 128x128 pixels for uniformity.

## Installation
To get started, clone the repository and install the required packages:

##Run the program
-run 'data_loader.py'
-run 'split_data.py'
-run 'train_sklearn.py'
-run 'evaluate_sklearn.py'
