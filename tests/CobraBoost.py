import numpy as np
from math import log
from itertools import compress
import random

from cobraclassifier import classifier_cobra as cobra
from cobraclassifier import edited_knn, near_miss_v1, near_miss_v2, near_miss_v3, tomek_link, condensed_knn, knn_und

class CobraBoost:
    def __init__(self, X, y, machines, undersampling_method):
        self.weight_update = 0
        self.X = X
        self.y = y
        self.model = cobra.CobraClassifier(machines = machines)
        self.majority_class_label = int(sum(y) > 0.5 * len(y))
        self.undersampling_method = undersampling_method

        # initialize weight
        self.weight = []
        self.init_w = 1.0 / len(self.X)
        for i in range(len(self.X)):
            self.weight.append(self.init_w)


    def learn_parameters(self, iterations):
        verdict = self.undersampling_method.undersample(self.X, self.y, self.majority_class_label)
        X_undersampled, y_undersampled = self.X[verdict, :], self.y[verdict]
        
        self.model.fit(X_undersampled, y_undersampled)
        
        for t in range(iterations):
            print("[Testing]: Executing the iteration - {} of CobraBoost".format(t + 1))
            flag = self.y != self.model.predict(self.X)
            loss = sum(list(compress(self.weight, flag)))
            
            # calculate loss equal to: weight sum of classifications:
            # >> correct classifications = 0
            # >> misclassifications = 1

            self.weight_update = loss / (1-loss)
            
            # update weights if there is a loss
            if loss > 0: 
                for i in range(len(self.weight)):
                    if self.y[i] == self.model.predict(self.X[i].reshape(1, -1)):
                        self.weight[i] = self.weight[i] * (loss / (1 - loss))

            # calculate total sum for normalization
            sum_weight = sum(self.weight) 

            # use total sum to normalize sum of weights to 1.0
            self.weight = [w / sum_weight for w in self.weight]


    def predict(self, test_data):
        n = len(test_data)
        predicted_labels = np.zeros(n)

        for i in range(n):
            positive_score, negative_score = 0, 0
            
            if self.model.predict(test_data[i].reshape(1, -1)) == 1:
                positive_score += log(1/self.weight_update)
            else:
                negative_score += log(1/self.weight_update)
            
            if negative_score <= positive_score:
                predicted_labels[i] = 1

        return predicted_labels