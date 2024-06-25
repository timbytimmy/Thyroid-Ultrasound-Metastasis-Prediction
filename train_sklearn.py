# train_sklearn_model.py
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Load the data splits
data = np.load('data_splits.npz')
X_train, X_val, X_test, y_train, y_val, y_test = data['X_train'], data['X_val'], data['X_test'], data['y_train'], data[
    'y_val'], data['y_test']

# Flatten the images
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_val_flat = X_val.reshape(X_val.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

# Train a Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_flat, y_train)

# Validate the model
y_val_pred = clf.predict(X_val_flat)
val_accuracy = accuracy_score(y_val, y_val_pred)
print(f'Validation Accuracy: {val_accuracy}')

# Save the model


joblib.dump(clf, 'thyroid_rf_model.joblib')
