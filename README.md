# COBRA for Classification tasks on Imbalanced Data
Implementation of the COBRA model and various undersampling algorithms to handle the Class Imbalance Problem as project (group Cobra-6) for the course MA691: Advanced Statistical Algorithms (Instructor: Dr. Arabin Kumar Dey), taken in Monsoon Semester 2021 at IIT Guwahati.   
A new hybrid algorithm ``CobraBoost`` is implemented which combines COBRA with AdaBoost to efficiently deal with imbalanced data.


## Installation:
```python3
pip3 install imbcobra
```

## Dependencies:
- Python 3.4+
- numpy, scikit-learn, matplotlib, pandas


## Testing:
For testing the code locally, execute following steps:
```console
home@:~$ git clone https://github.com/vishishtpriyadarshi/imbcobra
home@:~$ cd imbcobra
home@:/imbcobra$ pip3 install -r requirements.txt
home@:/imbcobra$ cd tests
home@:/imbcobra/tests$ python3 sample_test_1.py
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
import numpy as np
from imbcobra import classifier_cobra as cobra
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=1000, n_features=4,
                           n_informative=2, n_redundant=0,
                           random_state=0, shuffle=False)

model = cobra.CobraClassifier(machines = ['knn', 'logistic_regression', 'svm', 'naive_bayes', 'ridge', 'random_forest'])
model.fit(X, y)
model.predict(np.array([[0, 0, 0, 0]]))
```

### 2. CobraBoost -

## Example:
```python3
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

model.learn_parameters(iterations=5)
model.predict(np.array([[0, 0, 0, 0]]))
```

### 3. Undersampling algorithms - 
Following undersampling algorithms are available:

| Option | Algorithm |
| --- | ----------- |
| ```near_miss_v1``` | Near Miss - 1|
| ```near_miss_v2``` | Near Miss - 2 |
| ```near_miss_v3``` | Near Miss - 3 |
| ```condensed_knn``` | Condensed KNN |
| ```edited_knn``` | Edited KNN |
| ```knn_und``` | KNN Und |
| ```tomek_link``` | Tomek Links  |

## Example:
```python3
from imbcobra import edited_knn
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=1000, n_features=4,
                           n_informative=2, n_redundant=0,
                           random_state=0, shuffle=False)
                           
majority_class_label = int(sum(y) > 0.5 * len(y))
verdict = edited_knn.undersample(X, y, majority_class_label)
X_undersampled, y_undersampled = X[verdict, :], y_train[verdict]
```


## Results:
Comparison of negative recall for different algorithms across various datasets - 
![Plot](https://github.com/vishishtpriyadarshi/imbcobra/blob/main/utils/negative_recall.png?raw=true)


## Folder Structure:

```bash
???
????????? imbcobra                         # PyPi package 
???   ????????? ...
???
???
????????? report
???   ????????? Cobra-6 Project Report       # Project report for the complete analysis
???
???
????????? tests
???   ????????? cobra
???   ???   ????????? classifiercobra.py       # Definition of Classifier Cobra class
???   ???   ????????? CobraBoost.py            # Definition of class for the Boosting algorithm based on COBRA and AdaBoost
???   ???
???   ????????? datasets                     # Dataset for the classification tasks available at https://archive.ics.uci.edu/ml/index.php
???   ???   ????????? abalone.data
???   ???   ????????? allbp.data
???   ???   ????????? car.data
???   ???   ????????? ecoli.csv
???   ???   ????????? nursery.data
???   ???   ????????? winequality-red.csv
???   ???   ????????? winequality-white.csv
???   ???
???   ????????? models
???   ???   ????????? adaboost.py              # Implementation of AdaBoost algorithm from scratch
???   ???   ????????? logistic_regression.py   # Logistic Regression helper function to train models and get predictions easily
???   ???   ????????? parent_model.py          # Parent function to execute different models and handle pre-processing tasks
???   ???
???   ????????? undersampling_algorithms     # Implementation of Undersampling Algorithms
???   ???   ????????? __init__.py
???   ???   ????????? condensed_knn.py
???   ???   ????????? edited_knn.py
???   ???   ????????? knn_und.py
???   ???   ????????? near_miss_v1.py
???   ???   ????????? near_miss_v2.py
???   ???   ????????? near_miss_v3.py
???   ???   ????????? tomek_link.py
???   ???
???   ????????? sample_test_1.py             # Test the execution of COBRA and undersampling algorithms
???   ????????? sample_test_2.py             # Test the execution of CobraBoost
???   ????????? sample_test_3.py             # Test the package
???
???
????????? utils
???   ????????? results                      # Results for the execution of COBRA and undersampling algorithms
???   ???   ????????? ...          
???   ????????? MA691_Project.ipynb          # Preliminary work in the form of ipynb notebook
??? 
????????? ...
```


## References:
- G. Biau, A. Fischer, B. Guedj and J. D. Malley (2016), COBRA: A combined regression strategy, Journal of Multivariate Analysis.
- B. Guedj and B. Srinivasa Desikan (2018). Pycobra: A Python Toolbox for Ensemble Learning and Visualisation. Journal of Machine Learning Research, vol. 18 (190), 1--5.


## Disclaimer:
 This work is for learning purpose only.  The work cannot be used for publication or as commercial products etc without the mentor???s consent.
