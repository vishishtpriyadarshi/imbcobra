import numpy as np
from math import log
from itertools import compress

from cobraclassifier import classifier_cobra as cobra
from cobraclassifier import edited_knn, near_miss_v1, near_miss_v2, near_miss_v3, tomek_link, condensed_knn, knn_und


class CobraBoost:
    def __init__(self, X, y, machines, undersampling_method):
        self.X = X
        self.y = y

        self.model = cobra(machines = machines)
        self.majority_class_label = int(sum(y) > 0.5 * len(y))
        self.undersampling_method = undersampling_method

        self.weight_update = 0
        self.init_w = 1.0 / len(self.X)
        self.weight = np.full(len(self.X), self.init_w)
        

    def learn_parameters(self, iterations):
        verdict = self.undersampling_method.undersample(self.X, self.y, self.majority_class_label)
        X_undersampled, y_undersampled = self.X[verdict, :], self.y[verdict]
    
        for t in range(iterations):
            print("[Testing]: Executing the iteration - {} of CobraBoost".format(t + 1))
            
            self.model.fit(X_undersampled, y_undersampled, sample_weight = self.weight[verdict])
        
            flag = self.y != self.model.predict(self.X)
            loss = sum(list(compress(self.weight, flag)))
            
            alpha = loss / (1 - loss)  

            if alpha <= 0:
                alpha = 0.0000001
            else:
                try:
                    alpha_hat = 0.5 * (np.log(1 - loss) - np.log(loss))
                except:
                    alpha_hat = 0
                    
                self.weight = self.weight * np.exp(-alpha_hat * self.y * self.model.predict(self.X))
                self.weight = self.weight / self.weight.sum()

            self.weight_update = alpha
            

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