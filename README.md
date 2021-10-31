# COBRA for Classification tasks on Imbalanced Data

## Installation
```python3
pip3 install cobraclassifier
```

## Dependencies:
- Python 3.4+
- numpy, scikit-learn, matplotlib, pandas


## Testing:
``` 
git clone https://github.com/vishishtpriyadarshi/MA691-COBRA-6
cd MA691-COBRA-6
pip3 install -r requirements.txt
cd tests
python3 execute.py 
```

## Usage:
### 1. Cobra Classifier -
Following machines can be specified while initialising the model:

| Option | Classifier |
| --- | ----------- |
| ```knn``` | K Nearest Neighbors Classifier |
| ```random_forest``` | Random Forest |
| ```logistic_regression``` | Logistic Regression |
| ```svm``` | Support Vector Machine |
| ```decision_trees``` | Decision Trees |
| ```naive_bayes``` | Gaussian Naive Bayes |
| ```stochastic_gradient_descent``` | Stochastic Gradient Descent  |
| ```ridge``` | Ridge Classifer |

## Example:
```python3
from cobraclassifier import classifier_cobra as cobra
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=1000, n_features=4,
...                        n_informative=2, n_redundant=0,
...                        random_state=0, shuffle=False)
model = cobra.CobraClassifier(machines = ['knn', 'logistic_regression', 'svm', 'naive_bayes', 'ridge', 'random_forest'])
model.fit(X, y)
model.predict(np.array([[0, 0, 0, 0]]))
```

### 2. Undersampling algorithms - 
Following undersampling algorithms are available:
1. Near Miss Algorithm - v1, 2 and 3
2. Condensed KNN
3. Edited KNN
4. KNN Und
5. Tomek Links

## Example:
```python3
from cobraclassifier import edited_knn
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=1000, n_features=4, n_informative=2, n_redundant=0, random_state=0, shuffle=False)
majority_class_label = int(sum(y) > 0.5 * len(y))
verdict = edited_knn.undersample(X, y, majority_class_label)
X_undersampled, y_undersampled = X[verdict, :], y_train[verdict]
```

## References:
- G. Biau, A. Fischer, B. Guedj and J. D. Malley (2016), COBRA: A combined regression strategy, Journal of Multivariate Analysis.
- B. Guedj and B. Srinivasa Desikan (2018). Pycobra: A Python Toolbox for Ensemble Learning and Visualisation. Journal of Machine Learning Research, vol. 18 (190), 1--5.
