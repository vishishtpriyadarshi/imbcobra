import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import datasets

# from cobraclassifier import classifier_cobra, parent_model
# from cobraclassifier import near_miss_v1, near_miss_v2, near_miss_v3, knn_und, edited_knn, condensed_knn, tomek_link

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics

from CobraBoost import CobraBoost
from undersampling_algorithms import near_miss_v1, near_miss_v2, near_miss_v3, knn_und, edited_knn, condensed_knn, tomek_link
from cobra import classifier_cobra

import warnings
warnings.filterwarnings("ignore")

def prepare_data(seed, choice=1):
  if choice == 1:
    """ Dataset - 1 """
    print("==================  Executing Random dataset  ==================")
    # Ref - https://scikit-learn.org/stable/auto_examples/datasets/plot_random_dataset.html#sphx-glr-auto-examples-datasets-plot-random-dataset-py
    X, y = datasets.make_classification(n_samples = 1000, n_classes = 2, weights = [0.2, 0.8], class_sep = 0.9, 
                                  n_features = 5, n_redundant = 1, n_informative = 3, n_clusters_per_class = 1, random_state = seed)
    majority_class_label = 1
  elif choice == 2:
    """ Dataset - 2 """
    print("==================  Red Wine Quality dataset  ==================")
    dataset = pd.read_csv('./datasets/winequality-red.csv', sep=';')

    def reviews(row):
      if row['quality'] > 7:
          return 1
      else: 
          return 0

    dataset['reviews'] = dataset.apply(reviews, axis=1)
    dataset['reviews'].value_counts()

    features = list(dataset.columns)[:-1]
    target = 'reviews'
    X = np.asarray(dataset[features])
    y = np.asarray(dataset[target])

    majority_class_label = int(sum(y) > 0.5 * len(y))
    print("[Testing]: Majority class = ", majority_class_label, "\tSum of Target variable = ", sum(y), "\tLength of Target variable = ", len(y))
    # print(dataset.describe())

  elif choice == 3:
    """ Dataset - 3 """
    print("==================  White Wine Quality dataset  ==================")
    dataset = pd.read_csv('./datasets/winequality-white.csv', sep=';')

    def reviews(row):
      if row['quality'] > 7:
          return 1
      else: 
          return 0

    dataset['reviews'] = dataset.apply(reviews, axis=1)
    dataset['reviews'].value_counts()

    features = list(dataset.columns)[:-1]
    target = 'reviews'
    X = np.asarray(dataset[features])
    y = np.asarray(dataset[target])

    majority_class_label = int(sum(y) > 0.5 * len(y))
    print("[Testing]: Majority class = ", majority_class_label, "\tSum of Target variable = ", sum(y), "\tLength of Target variable = ", len(y))
    # print(dataset.describe())

  elif choice == 4:
    """ Dataset - 4 """
    print("==================  Car evaluation dataset  ==================")
    # Ref - https://archive.ics.uci.edu/ml/datasets/car+evaluation
    dataset = pd.read_csv('./datasets/car.data', header=None, sep=',')
    
    """ Pre-processing """
    def create_target(row, val='vgood'):
      if row[6] == val:
          return 1
      else: 
          return 0

    def encode_attribute_1(row):
      if row[0] == 'vhigh':   return 1
      elif row[0] == 'high':  return 2
      elif row[0] == 'med':   return 3
      else:                   return 4
    
    def encode_attribute_2(row):
      if row[1] == 'vhigh':   return 1
      elif row[1] == 'high':  return 2
      elif row[1] == 'med':   return 3
      else:                   return 4

    def encode_attribute_3(row):
      if row[2] == '5more':   return 5
      else:                   return row[2]

    def encode_attribute_4(row):
      if row[3] == 'more':    return 5
      else:                   return row[3]

    def encode_attribute_5(row):
      if row[4] == 'small':   return 1
      elif row[4] == 'med':   return 2
      else:                   return 3

    def encode_attribute_6(row):
      if row[5] == 'low':     return 1
      elif row[5] == 'med':   return 2
      else:                   return 3

    dataset[0] = dataset.apply(encode_attribute_1, axis=1)
    dataset[1] = dataset.apply(encode_attribute_2, axis=1)
    dataset[2] = dataset.apply(encode_attribute_3, axis=1)
    dataset[3] = dataset.apply(encode_attribute_4, axis=1)
    dataset[4] = dataset.apply(encode_attribute_5, axis=1)
    dataset[5] = dataset.apply(encode_attribute_6, axis=1)
    dataset['target'] = dataset.apply(create_target, axis=1)
    
    """"""

    features = list(dataset.columns)[:-2]
    target = 'target'
    X = np.asarray(dataset[features])
    y = np.asarray(dataset[target])

    majority_class_label = int(sum(y) > 0.5 * len(y))
    print("[Testing]: Majority class = ", majority_class_label, "\tSum of Target variable = ", sum(y), "\tLength of Target variable = ", len(y))
    # print(dataset.describe())
    # print(dataset.head(10))

  elif choice == 5:
    """ Dataset - 5 """
    print("==================  Ecoli dataset  ==================")
    # Ref - https://archive.ics.uci.edu/ml/datasets/ecoli, https://www.kaggle.com/kannanaikkal/ecoli-uci-dataset
    dataset = pd.read_csv('./datasets/ecoli.csv')
    
    def create_target(row, val='imU'):
      if row['SITE'] == val:
          return 1
      else: 
          return 0

    dataset = dataset.drop('SEQUENCE_NAME', 1)
    # for s in set(dataset['SITE']):
    #   print(s, " = ", dataset['SITE'].str.count(s).sum())

    dataset['target'] = dataset.apply(create_target, axis=1)
    
    features = list(dataset.columns)[:-2]
    target = 'target'
    X = np.asarray(dataset[features])
    y = np.asarray(dataset[target])

    majority_class_label = int(sum(y) > 0.5 * len(y))
    print("[Testing]: Majority class = ", majority_class_label, "\tSum of Target variable = ", sum(y), "\tLength of Target variable = ", len(y))
    # print(dataset.describe())
    # print(dataset.head(10))

  elif choice == 6:
    """ Dataset - 6 """
    # Ref - https://archive.ics.uci.edu/ml/datasets/abalone
    print("==================  Abalone dataset  ==================")
    dataset = pd.read_csv('./datasets/abalone.data', header=None, sep=',')
    
    def create_target(row, val=20):
      if row[8] >= val:
          return 1
      else: 
          return 0

    def encode_attribute(row):
      if row[0] == 'M':     return 0
      elif row[0] == 'F':   return 1
      else:                 return 2

    dataset[0] = dataset.apply(encode_attribute, axis = 1)
    dataset['target'] = dataset.apply(create_target, axis=1)
    dataset['target'].value_counts()

    features = list(dataset.columns)[:-2]
    target = 'target'
    X = np.asarray(dataset[features])
    y = np.asarray(dataset[target])

    majority_class_label = int(sum(y) > 0.5 * len(y))
    print("[Testing]: Majority class = ", majority_class_label, "\tSum of Target variable = ", sum(y), "\tLength of Target variable = ", len(y))
    # print(dataset.describe())
    # print(dataset.head(10))
    # print(dataset[8].value_counts())

  elif choice == 7:
    """ Dataset - 7 """
    # Ref - https://archive.ics.uci.edu/ml/datasets/nursery
    print("==================  Nursery dataset  ==================")
    dataset = pd.read_csv('./datasets/nursery.data', header=None, sep=',')
    
    def create_target(row, val='very_recom'):
      if row[8] == val:
          return 1
      else: 
          return 0

    def encode_attributes(original_dataframe, feature_to_encode):
      dummies = pd.get_dummies(original_dataframe[[feature_to_encode]])
      res = pd.concat([original_dataframe, dummies], axis=1)
      res = res.drop([feature_to_encode], axis=1)
      return res

    features_to_encode = [0, 1, 2, 3, 4, 5, 6, 7]
    for feature in features_to_encode:
      dataset = encode_attributes(dataset, feature)

    dataset['target'] = dataset.apply(create_target, axis=1)

    features = list(dataset.columns)[1:-1]
    target = 'target'
    X = np.asarray(dataset[features])
    y = np.asarray(dataset[target])

    majority_class_label = int(sum(y) > 0.5 * len(y))
    print("[Testing]: Majority class = ", majority_class_label, "\tSum of Target variable = ", sum(y), "\tLength of Target variable = ", len(y))
    # print(dataset.describe())
    # print(dataset.head(10))
    # for s in set(dataset[8]):
    #   print(s, " = ", dataset[8].str.count(s).sum())

  else:
    print("No dataset available")

  # plt.scatter(X[:, 0], X[:, 1], marker = '.', c = y)
  # plt.show()
  # assert np.any(np.isnan(dataset)) == False
  
  return X, y, majority_class_label


def execute_cobraboost(X, y, num_splits, seed, machines, undersampling_method = knn_und, boosting_iterations = 4):
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
    
    # Model Fitting & Predictions on test dataset
    model = CobraBoost(X_train, y_train, machines, undersampling_method)
    model.learn_parameters(boosting_iterations)
    y_pred = model.predict(X_test)

    # Evaluation Metrics
    accuracy = metrics.accuracy_score(y_test, y_pred)
    precision, recall, F1_score, _ = metrics.precision_recall_fscore_support(y_test, y_pred, beta = 1.0, average = 'macro')
    metrics_list.append([accuracy, precision, recall, F1_score])

    classification_report = metrics.classification_report(y_test, y_pred, target_names = ['class 0', 'class 1'], output_dict = True)
    class0_report, class1_report = classification_report['class 0'], classification_report['class 1']
    class0_metrics_list.append([class0_report['precision'], class0_report['recall'], class0_report['f1-score']])
    class1_metrics_list.append([class1_report['precision'], class1_report['recall'], class1_report['f1-score']])

    iterations += 1
  
  metrics_list = np.mean(metrics_list, axis = 0)
  class0_metrics_list = np.mean(class0_metrics_list, axis = 0)
  class1_metrics_list = np.mean(class1_metrics_list, axis = 0)

  print("\n---------------  Cross-validated Evaluation Metrics  ---------------\n")
  print("Accuracy \t= \t", metrics_list[0])
  print("Precision \t= \t", metrics_list[1])
  print("Recall \t\t= \t", metrics_list[2])
  print("Negative Recall = \t", class1_metrics_list[2])
  print("F1 score \t= \t", 2 * metrics_list[1] * metrics_list[2] / (metrics_list[1] + metrics_list[2]))


def main():
    # ================  Menu  ================
    print("==================  Available Options for Datasets:  ==================")
    print("Choose dataset -")
    print("\nRandomly generated dataset\t-\t1")
    print("Red Wine Quality\t\t-\t2")
    print("White Wine Quality\t\t-\t3")
    print("Car Evaluation\t\t\t-\t4")
    print("Ecoli\t\t\t\t-\t5")
    print("Abalone\t\t\t\t-\t6")
    print("Nursery\t\t\t\t-\t7\n\n")
    ch = int(input("Enter your choice (between 1 to 7): "))
    # ========================================

    num_splits, seed = 2, 32
    X, y, _ = prepare_data(seed, choice=ch)
    machines = ['knn', 'logistic_regression', 'svm', 'naive_bayes', 'ridge', 'random_forest']
    execute_cobraboost(X, y, num_splits, seed, machines, undersampling_method = near_miss_v3, boosting_iterations = 4)


if __name__ == "__main__":
    main()