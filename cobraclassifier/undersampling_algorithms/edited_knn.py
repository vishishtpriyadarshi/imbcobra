import numpy as np
from sklearn.neighbors import NearestNeighbors


def undersample(X, y, majority_class, k = 5, knn_algorithm = 'auto', knn_metric = 'euclidean'):
  n, _ = X.shape
  verdict = np.ones(n, dtype=bool)

  # Step - 1: Find k nearest neighbors of all the data points 
  nearest_neighbors = NearestNeighbors(n_neighbors = k, algorithm = knn_algorithm, metric = knn_metric).fit(X)
  _, nearest_neighbors_idx = nearest_neighbors.kneighbors(X)

  # Step - 2: Delete elements from Majority Class
  for i in range(n):
    if(y[i] == majority_class):
      tmp = 0
      for j in nearest_neighbors_idx[i]:
        tmp += y[j]
      tmp/=3
      if(tmp<0.6):
        verdict[i] = False
    else:
      tmp = 0
      for j in nearest_neighbors_idx[i]:
        tmp += y[j]
      tmp/=3
      if(tmp>0.6):    
        for j in nearest_neighbors_idx[i]:
          if(y[j] == majority_class):
            verdict[j] == False

  return verdict            