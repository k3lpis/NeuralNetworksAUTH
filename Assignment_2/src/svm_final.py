import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve 
from sklearn.decomposition import PCA
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Load CIFAR-10 dataset using TensorFlow
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Choose two classes for binary classification (e.g., cats and dogs)
class1, class2 = 3, 5  

# Filter data for the selected classes in both training and test sets
train_class_indices = np.where((y_train == class1) | (y_train == class2))[0]
test_class_indices = np.where((y_test == class1) | (y_test == class2))[0]

X_train_binary = X_train[train_class_indices]
y_train_binary = y_train[train_class_indices]
X_test_binary  = X_test[test_class_indices]
y_test_binary = y_test[test_class_indices] 

# Flatten the image data
X_train_binary = X_train_binary.reshape(X_train_binary.shape[0], -1)
X_test_binary = X_test_binary.reshape(X_test_binary.shape[0], -1)

# Standardize features by removing the mean and scaling to unit variance
scaler = StandardScaler()
X_train_binary = scaler.fit_transform(X_train_binary)
X_test_binary = scaler.transform(X_test_binary)

#flatten the target variable
y_train_binary = y_train_binary.ravel()
y_test_binary = y_test_binary.ravel()

#Apply PCA for dimensionality reduction
pca = PCA(n_components=0.90)  
X_train_binary = pca.fit_transform(X_train_binary)
X_test_binary = pca.transform(X_test_binary)

# Create an SVM classifier
svm_classifier2 = SVC(kernel='poly', C=1.0, random_state=42)
svm_classifier3 = SVC(kernel='rbf', C=1.0, random_state=42)
print(f"The gamma value used in the svm_classifier3 is {svm_classifier3.gamma}.")


start_time = time.time()
svm_classifier2.fit(X_train_binary, y_train_binary)
end_time = time.time()
print(f"Training time (polynomial kernel): {end_time - start_time:.2f} seconds")

start_time = time.time()
svm_classifier3.fit(X_train_binary, y_train_binary)
end_time = time.time()
print(f"Training time (rbf kernel): {end_time - start_time:.2f} seconds")

print(f"The gamma value used in the svm_classifier3 is {svm_classifier3.gamma}.")
# Make predictions on the test set
y_pred2 = svm_classifier2.predict(X_test_binary)
y_pred3 = svm_classifier3.predict(X_test_binary)

# Evaluate the accuracy
accuracy2 = accuracy_score(y_test_binary, y_pred2)
accuracy3 = accuracy_score(y_test_binary, y_pred3)
print(f"Accuracy with polynomial kernel: {accuracy2 * 100:.2f}%")
print(f"Accuracy with radial basis function (rbf): {accuracy3 * 100:.2f}%")

print(f"The gamma value used in the svm_classifier3 is {svm_classifier3.gamma}.")
# Grid Search
parameters_grid = {'kernel': ['poly'],
              'C': [0.1, 1, 10], 
                   'degree' : [1, 2, 3, 4] }
grid_search = GridSearchCV(SVC(random_state=42), parameters_grid, cv=3, scoring='accuracy', verbose = 2)

# Measure grid search time
start_time = time.time()
grid_search.fit(X_train_binary, y_train_binary)
end_time = time.time()

# Get the best hyperparameters
best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)

# Get the best model
best_svm_classifier = grid_search.best_estimator_

# Evaluate the best model on the test set
y_pred_best = best_svm_classifier.predict(X_test_binary)
accuracy_best = accuracy_score(y_test_binary, y_pred_best)
print(f"Best Model Accuracy on Test Set: {accuracy_best * 100:.2f}%")

print(f"Grid Search time: {end_time - start_time:.2f} seconds")

# Plotting the learning curves
train_sizes, train_scores, test_scores = learning_curve(svm_classifier3, X_train_binary, y_train_binary, cv=5)
train_scores_mean = np.mean(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)

plt.figure()
plt.plot(train_sizes, train_scores_mean, label='Training score')
plt.plot(train_sizes, test_scores_mean, label='Cross-validation score')
plt.xlabel('Training examples')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Confusion Matrix
conf_matrix = confusion_matrix(y_test_binary, y_pred_best)
print("Confusion Matrix:")
print(conf_matrix)

# Plotting the confusion matrix
class_names = [f"Class {class1}", f"Class {class2}"]
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Visualizing some right and wrong examples
correct_indices = np.where(y_test_binary == y_pred_best)[0]
incorrect_indices = np.where(y_test_binary != y_pred_best)[0]

# Map class labels to indices
label_to_index = {class1: 0, class2: 1}

# Display 10 correct examples
plt.figure(figsize=(20, 8))
for i, idx in enumerate(correct_indices[:10]):
    plt.subplot(2, 10, i + 1)
    plt.imshow(X_test[test_class_indices][idx])  # Use original image data
    plt.title(f'True: {class_names[label_to_index[y_test_binary[idx]]]}\nPred: {class_names[label_to_index[y_pred_best[idx]]]}')
    plt.axis('off')

# Display 10 incorrect examples
for i, idx in enumerate(incorrect_indices[:10]):
    plt.subplot(2, 10, i + 11)
    plt.imshow(X_test[test_class_indices][idx])  # Use original image data
    plt.title(f'True: {class_names[label_to_index[y_test_binary[idx]]]}\nPred: {class_names[label_to_index[y_pred_best[idx]]]}')
    plt.axis('off')

plt.show()
