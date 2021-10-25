from sklearn.utils import shuffle

from sklearn.linear_model import LogisticRegression, SGDClassifier, RidgeClassifier
from sklearn import neighbors, tree, svm
from sklearn.base import BaseEstimator
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

import numpy as np
import math


class CobraClassifier:
    def __init__(self, seed=42, epsilon=0.001, threshold=0.5, machines=None):
        self.seed = seed
        self.epsilon = epsilon
        self.threshold = threshold
        self.machines = machines


    def fit(self, X, y, X_k=None, y_k=None, X_l=None, y_l=None, flag=False):
        # if flag == True => X_k values are defined
        self.X, self.y = X, y
        self.X_k, self.y_k = X_k, y_k
        self.X_l, self.y_l = X_l, y_l
        
        self.machine_estimators = {}

        if flag == False:
            self.generate_training_data()

        # Train machine estimators on the training data (D_k)
        self.setup_machines()
        self.execute_machines()

        return self


    def predict_helper(self, X, alpha):
        res = {}
        for m in self.machines:
            predicted_label = self.machine_estimators[m].predict(X)
            res[m] = {}

            for idx in range(len(self.X_l)):
                if math.fabs(self.machine_predictions[m][idx] - predicted_label) <= self.epsilon:
                    res[m][idx] = 1
                else:
                    res[m][idx] = 0
        
        filtered_points = []
        for idx in range(0, len(self.X_l)):
            sum = 0
            for m in res:
                if res[m][idx] == 1:
                    sum += 1
                if sum >= alpha:
                    filtered_points.append(idx)
                    break

        if len(filtered_points) == 0:
            return 0

        score = 0
        for idx in filtered_points:
            score += self.y_l[idx]
        score = score / len(filtered_points)
        
        final_label = 1 if score >= self.threshold else 0
        return final_label


    def predict(self, X, alpha=None):
        n = len(X)
        predicted_labels = np.zeros(n)
        
        if alpha is None:
            alpha = len(self.machines)

        for i in range(n):
            predicted_labels[i] = self.predict_helper(X[i].reshape(1, -1), alpha)
        
        return predicted_labels


    def set_epsilon(self):
        pass


    def setup_machines(self):
        for m in self.machines:
            if m == 'knn':
                self.machine_estimators[m] = neighbors.KNeighborsClassifier().fit(self.X_k, self.y_k)
            elif m == 'random_forest':
                self.machine_estimators[m] = RandomForestClassifier(random_state=self.seed).fit(self.X_k, self.y_k)
            elif m == 'logistic_regression':
                self.machine_estimators[m] = LogisticRegression(random_state=self.seed).fit(self.X_k, self.y_k)
            elif m == 'svm':
                self.machine_estimators[m] = svm.SVC().fit(self.X_k, self.y_k)
            elif m == 'decision_trees':
                self.machine_estimators[m] = tree.DecisionTreeClassifier().fit(self.X_k, self.y_k)
            elif m == 'naive_bayes':
                self.machine_estimators[m] = GaussianNB().fit(self.X_k, self.y_k)
            elif m == 'stochastic_gradient_decision':
                self.machine_estimators[m] = SGDClassifier().fit(self.X_k, self.y_k)
            elif m == 'ridge':
                self.machine_estimators[m] = RidgeClassifier().fit(self.X_k, self.y_k)

        return self


    def execute_machines(self):
        self.machine_predictions = {}
  
        for m in self.machines:
            self.machine_predictions[m] = self.machine_estimators[m].predict(self.X_l)

        return self
    

    def generate_training_data(self, k=None, l=None):
        """ 
        Splits the data into training (D_k) and testing part (D_l) for execution of models as specified in the COBRA paper
        """

        if k is None or l is None:
            n = len(self.X)
            k = int(3*n/4)
            l = int(n/4)

        self.X_k, self.y_k = self.X[ : k], self.y[ : k]
        self.X_l, self.y_l = self.X[k : ], self.y[k : ]
        
        return self


def execute_cobra(X_train, y_train, X_test):
  print("[Executing]: Running Cobra Model ...\n")

  # Model Fitting
  model = CobraClassifier(machines = ['knn', 'logistic_regression', 'svm', 'naive_bayes', 'ridge'])  
  model.fit(X_train, y_train)

  # Predictions on test dataset
  y_pred = model.predict(X_test)

  return y_pred