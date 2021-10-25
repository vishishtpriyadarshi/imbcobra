import numpy as np
from sklearn.neighbors import NearestNeighbors


def undersample(X, y, majority_class, k = 50, t = 2, knn_algorithm = 'auto', knn_metric = 'euclidean'):
  n, _ = X.shape
  verdict = np.ones(n, dtype=bool)

  # Step - 1: Find k nearest neighbors of all the data points 
  nearest_neighbors = NearestNeighbors(n_neighbors = k, algorithm = knn_algorithm, metric = knn_metric).fit(X)
  _, nearest_neighbors_idx = nearest_neighbors.kneighbors(X)

  for i in range(n):
    # Step - 2: Identify the class label of X_i
    class_label = y[i]

    # Step - 3: Proceed only with majority class data points
    if class_label != majority_class:
      continue

    # Step - 4: Identify the count of minority class neighbors for the X_i
    minority_class_neighbors_count = 0
    neighbors_class_label = y[nearest_neighbors_idx[i]]

    for j in neighbors_class_label:
      if j != majority_class:
        minority_class_neighbors_count += 1

    # Step - 5: Mark the data point as False if count of minority class neighbors >= t
    if minority_class_neighbors_count >= t:
      verdict[i] = False

  return verdict