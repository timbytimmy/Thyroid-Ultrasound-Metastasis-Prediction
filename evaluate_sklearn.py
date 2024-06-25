# evaluate_sklearn_model.py
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# Load the data splits
data = np.load('data_splits.npz')
X_test, y_test = data['X_test'], data['y_test']

# Flatten the images
X_test_flat = X_test.reshape(X_test.shape[0], -1)

# Load the trained model
clf = joblib.load('thyroid_rf_model.joblib')

# Evaluate the model
y_test_pred = clf.predict(X_test_flat)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f'Test Accuracy: {test_accuracy}')

# Confusion Matrix
cm = confusion_matrix(y_test, y_test_pred)

plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
