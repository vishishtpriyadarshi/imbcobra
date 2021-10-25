import numpy as np
from sklearn.neighbors import NearestNeighbors


def undersample(X, y, majority_class, k = 3, ratio = 4.5, knn_algorithm = 'auto', knn_metric = 'euclidean'):
  n, _ = X.shape
  verdict = np.ones(n, dtype=bool)

  # Step - 1: Find k farthest neighbors (in minority class) of all the data points
  X_minority = []
  for idx, _ in enumerate(X):
    if y[idx] != majority_class:
      X_minority.append(X[idx])
      
  distances = []
  for idx, maj_val in enumerate(X):
    if idx % 500 == 0:
      print("[Executing]: Processing data point - {} while undersampling...".format(idx + 1))

    if y[idx] != majority_class:
      distances.append([0] * k)    # To allow executing 'np.mean(distances, axis = 1)'
    else:
      current_distances = []
      for min_val in X_minority:
        current_distances.append(np.linalg.norm(maj_val - min_val))
      current_distances.sort(reverse = True)

      distances.append(current_distances[: min(k, len(X_minority))])
  
  # Step - 2: Compute the average distance
  distances = np.mean(distances, axis = 1)
  
  # Step - 3: Select the points from majority class depending on the 'ratio' value
  X_majority_info = []  # stores a 2D value, where val at idx1 is the index in the original data and val at idx2 is the mean distance
  for i in range(n):
    class_label = y[i]
    if class_label == majority_class:
      X_majority_info.append([i, distances[i]])

  majority_size = len(X_majority_info)
  X_majority_info.sort(key=lambda x: x[1])
  req_count = min(majority_size, (int)(ratio * len(X_minority)))
  
  print("[Testing]: Count of majority samples after undersampling vs Count of minority samples = ", req_count, "vs", len(X_minority), "or", req_count * 100/len(X_minority), "%")
  for i in range(req_count, majority_size):
    verdict[X_majority_info[i][0]] = False  

  return verdict