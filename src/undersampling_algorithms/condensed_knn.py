import numpy as np
from sklearn.neighbors import NearestNeighbors


def undersample(X, y, majority_class, knn_algorithm = 'auto', knn_metric = 'euclidean'):
  n, _ = X.shape
  verdict = np.zeros(n, dtype=bool)

  # Step - 1: Find k nearest neighbors (in minority class) of all the data points
  store, store_y, grabstore, grabstore_y, s_idx, g_idx = [], [], [], [], [], []
  X_minority, X_majority = [], []

  for idx, val in enumerate(X):
    if y[idx] != majority_class:
      X_minority.append(X[idx])
      s_idx.append(idx)
      store.append(X[idx])
      store_y.append(y[idx])
    else:
      X_majority.append(X[idx])
      g_idx.append(idx)
      grabstore.append(X[idx])
      grabstore_y.append(y[idx])
    
  # Step - 2: Compute the average distance
  cnt = 100
  while((cnt != 0) and (len(grabstore)!=0)):
    cnt = 0
    i = 0
    for sn, i in enumerate(np.array(grabstore)):
      tmp = []
      nearest_neighbors = NearestNeighbors(n_neighbors = 1, algorithm = knn_algorithm, metric = knn_metric).fit(store)
      distances, nearest_neighbors_idx = nearest_neighbors.kneighbors(i.reshape(1,-1))
      if(store_y[nearest_neighbors_idx[0][0]] != majority_class):
        store.append(i)
        store_y.append(grabstore_y[sn])
        s_idx.append(g_idx[sn])
        tmp.append(sn)
        cnt += 1
    for z in tmp:
      del(grabstore[z])
      del(grabstore_y[z])
      del(g_idx[z])   
  
  for i in s_idx:
    verdict[i] = True

  return verdict