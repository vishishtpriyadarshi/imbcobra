import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import datasets

# from cobraclassifier import classifier_cobra
# from cobraclassifier import near_miss_v1, near_miss_v2, near_miss_v3, knn_und, edited_knn, condensed_knn, tomek_link

from models.logistic_regression import logistic_regression
from models.adaboost import adaboost_classifier
from models.parent_model import execute_model
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

  num_splits, seed = 2, 14
  X, y, majority_class_label = prepare_data(seed, choice=ch)
  
  # models = [logistic_regression, adaboost_classifier, classifier_cobra.execute_cobra]
  # models = [classifier_cobra.execute_cobra]
  models = [logistic_regression, adaboost_classifier]
  # models = [logistic_regression]

  for m in models:
    print("\n\n#############################  MODEL -", m.__name__, "  #############################")
    print("\n=======================  Executing without undersampling  =======================")
    # parent_model.execute_model(X, y, num_splits, seed, m)
    execute_model(X, y, num_splits, seed, m)

    print("\n\n=======================  Executing with undersampling  =======================")
    # parent_model.execute_model(X, y, num_splits, seed, m, with_undersampling = True, majority_class = majority_class_label, undersampling_method = near_miss_v3)
    # execute_model(X, y, num_splits, seed, m, with_undersampling = True, majority_class = majority_class_label, undersampling_method = near_miss_v1)
    # execute_model(X, y, num_splits, seed, m, with_undersampling = True, majority_class = majority_class_label, undersampling_method = near_miss_v2)
    execute_model(X, y, num_splits, seed, m, with_undersampling = True, majority_class = majority_class_label, undersampling_method = near_miss_v3)
    # execute_model(X, y, num_splits, seed, m, with_undersampling = True, majority_class = majority_class_label, undersampling_method = condensed_knn)
    # execute_model(X, y, num_splits, seed, m, with_undersampling = True, majority_class = majority_class_label, undersampling_method = knn_und)
    # execute_model(X, y, num_splits, seed, m, with_undersampling = True, majority_class = majority_class_label, undersampling_method = edited_knn)
  

if __name__ == "__main__":
  main()