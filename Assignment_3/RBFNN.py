import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import rbf_kernel
import tensorflow as tf 
import matplotlib.pyplot as plt
import time 
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

# Load CIFAR-10 dataset using TensorFlow
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Preprocess the data
X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)
X_train = X_train.astype('float32') / 255.
X_test = X_test.astype('float32') / 255.

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Cluster the data using KMeans
kmeans = KMeans(n_clusters=100, random_state=42).fit(X_train)
centers = kmeans.cluster_centers_

# Compute the RBF kernel matrix
gamma = 1.0 / (2 * 100 ** 2)
K_train = rbf_kernel(X_train, centers, gamma=gamma)
K_test = rbf_kernel(X_test, centers, gamma=gamma)

# Train the RBF Neural Network
start_time = time.time()

rbf_nn = MLPClassifier(hidden_layer_sizes=(100,), activation='logistic', solver='lbfgs', max_iter=5000, random_state=42)
rbf_nn.fit(K_train, y_train)

end_time = time.time()

# Evaluate the RBF Neural Network
y_train_pred = rbf_nn.predict(K_train)
train_accuracy = accuracy_score(y_train, y_train_pred)
print(f'The accuracy of the RBF Neural Network on the training set is {train_accuracy:.2%}.')

y_test_pred = rbf_nn.predict(K_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f'The accuracy of the RBF Neural Network on the test set is {test_accuracy:.2%}.')
print(f"The RBF Neural Network took {end_time - start_time:.2f} seconds to train.")


# Get the predicted labels for the test set
y_pred = rbf_nn.predict(K_test)

# Get the indices of the misclassified images
misclassified_indices = [i for i, pred in enumerate(y_pred) if pred != y_test[i]]

# Plot the first 10 misclassified images
fig, axs = plt.subplots(2, 5, figsize=(15, 3))
axs = axs.ravel()
for i in range(5):
    axs[i].imshow(X_test[misclassified_indices[i]].reshape(32, 32, 3))
    axs[i].set_title(f"True label: {y_test[misclassified_indices[i]][0]}, Predicted label: {y_pred[misclassified_indices[i]]}")
    
# Get the indices of the correctly classified images
correct_indices = [i for i, pred in enumerate(y_pred) if pred == y_test[i]]

# Plot the first 10 correctly classified images
for i in range(5):
    axs[i+5].imshow(X_test[correct_indices[i]].reshape(32, 32, 3))
    axs[i+5].set_title(f"True label: {y_test[correct_indices[i]][0]}, Predicted label: {y_pred[correct_indices[i]]}")
plt.show()

# Define the range of values for the number of hidden neurons
param_grid = {'hidden_layer_sizes': [(50,), (100,), (150,), (200,), (300,)]}

# Create a grid search object
grid_search = GridSearchCV(rbf_nn, param_grid, cv=3, scoring = 'accuracy', verbose = 2)

# Fit the grid search object to the data
grid_search.fit(K_train, y_train)

# Print the results
print(f'The best number of hidden neurons is {grid_search.best_params_["hidden_layer_sizes"]}.')
print(f'The accuracy of the RBF Neural Network with {grid_search.best_params_["hidden_layer_sizes"]} hidden neurons on the test set is {grid_search.best_score_:.2%}.')
