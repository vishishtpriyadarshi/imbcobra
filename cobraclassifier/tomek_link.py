import numpy as np
from sklearn.neighbors import NearestNeighbors


def undersample(X, y, majority_class, knn_algorithm = 'auto', knn_metric = 'euclidean'):
  n, _ = X.shape
  verdict = np.ones(n, dtype=bool)

  # Step - 1: Find k nearest neighbors of all the data points (k = 1)
  nearest_neighbors = NearestNeighbors(n_neighbors = 1, algorithm = knn_algorithm, metric = knn_metric).fit(X)
  _, nearest_neighbors_idx = nearest_neighbors.kneighbors(X)

  for i in range(n):
    # Step - 2: Identify the class label of X_i
    class_label = y[i]

    # Step - 3: Proceed only with majority class data points
    if class_label != majority_class:
      continue

    # Step - 4: Remove the data point if it is part of a Tomek Link
    neighbor_class_label = y[nearest_neighbors_idx[i][0]]
    if neighbor_class_label != majority_class:
      verdict[i] = False

  return verdict