# COBRA for Classification tasks on Imbalanced Data

## Execution Steps:
``` 
git clone https://github.com/vishishtpriyadarshi/MA691-COBRA-6
cd MA691-COBRA-6
pip3 install -r requirements.txt
cd tests
python3 execute.py 
```

## Examples:
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

## Dependencies:
- Python 3.4+
- numpy, scikit-learn, matplotlib, pandas


## References:
- G. Biau, A. Fischer, B. Guedj and J. D. Malley (2016), COBRA: A combined regression strategy, Journal of Multivariate Analysis.
- B. Guedj and B. Srinivasa Desikan (2018). Pycobra: A Python Toolbox for Ensemble Learning and Visualisation. Journal of Machine Learning Research, vol. 18 (190), 1--5.
