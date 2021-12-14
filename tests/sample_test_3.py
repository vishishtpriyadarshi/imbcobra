import numpy as np
from imbcobra import cobra_boost as cobra
from imbcobra import edited_knn
from sklearn.datasets import make_classification


X, y = make_classification(n_samples=1000, n_features=4,
                            n_informative=2, n_redundant=0,
                            random_state=0, shuffle=False)

model = cobra.CobraBoost(X, y,
                        machines = ['knn', 'logistic_regression', 'svm', 'naive_bayes', 'ridge', 'random_forest'],
                        undersampling_method=edited_knn)

model.learn_parameters(iterations=2)
print(model.predict(np.array([[0, 0, 0, 0]])))