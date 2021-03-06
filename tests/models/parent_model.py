import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from imblearn.under_sampling import NearMiss

import sys
sys.path.insert(0, '..')
from undersampling_algorithms import *
import cobra.classifier_cobra


def execute_model(X, y, num_splits, seed, model, with_undersampling = False, majority_class = 0, undersampling_method = knn_und):
  K_folds = StratifiedKFold(n_splits = num_splits, shuffle = True, random_state = seed)
  metrics_list, class0_metrics_list, class1_metrics_list = [], [], []
  
  # Feature Scaling
  sc = StandardScaler()
  X = sc.fit_transform(X)

  iterations = 1
  for train_idx, test_idx in K_folds.split(X, y):
    print("\n****************  Executing iteration - {} of KFold Data split  ****************".format(iterations))

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # Execute Undersampling on training data
    if with_undersampling == True:
      print("[Testing]: Count of test data before Undersampling = ", X_train.shape[0])
      verdict = undersampling_method.undersample(X_train, y_train, majority_class)

      X_train = X_train[verdict, :]
      y_train = y_train[verdict]

      # In-buit near miss algorithm
      # nr = NearMiss()
      # X_train, y_train = nr.fit_resample(X_train, y_train)

      # Note: Be careful while plotting, make sure same features are being compared
      # plt.scatter(X_train[:, 0], X_train[:, 1], marker = '.', c = y_train)
      # plt.show()

      print("[Testing]: Count of test data after Undersampling = ", X_train.shape[0])

    # Model Fitting & Predictions on test dataset
    y_pred = model(X_train, y_train, X_test)

    # Evaluation Metrics
    accuracy = metrics.accuracy_score(y_test, y_pred)
    precision, recall, F1_score, _ = metrics.precision_recall_fscore_support(y_test, y_pred, beta = 1.0, average = 'macro')
    metrics_list.append([accuracy, precision, recall, F1_score])

    classification_report = metrics.classification_report(y_test, y_pred, target_names = ['class 0', 'class 1'], output_dict = True)
    class0_report, class1_report = classification_report['class 0'], classification_report['class 1']
    class0_metrics_list.append([class0_report['precision'], class0_report['recall'], class0_report['f1-score']])
    class1_metrics_list.append([class1_report['precision'], class1_report['recall'], class1_report['f1-score']])

    # print(metrics.confusion_matrix(y_test, y_pred))
    iterations += 1
  
  metrics_list = np.mean(metrics_list, axis = 0)
  class0_metrics_list = np.mean(class0_metrics_list, axis = 0)
  class1_metrics_list = np.mean(class1_metrics_list, axis = 0)
  

  print("\nMajority Class = Class ", majority_class)
  print("\n---------------  Cross-validated Evaluation Metrics  ---------------\n")
  print("Accuracy \t= \t", metrics_list[0])
  print("Precision \t= \t", metrics_list[1])
  print("Recall \t\t= \t", metrics_list[2])
  print("Negative Recall = \t", class1_metrics_list[2])
  print("F1 score \t= \t", 2 * metrics_list[1] * metrics_list[2] / (metrics_list[1] + metrics_list[2]))