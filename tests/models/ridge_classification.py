from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import GridSearchCV

def ridge_classification(X_train, y_train, X_test):
  print("[Executing]: Running Ridge Regression model ...\n")

  # Model Fitting
  model = RidgeClassifier(random_state=49)

  parameters = {'alpha':[1, 10]}
  final_model = GridSearchCV(model, parameters, scoring='neg_mean_squared_error', cv=5)
  final_model.fit(X_train, y_train)

  # Predictions on test dataset
  y_pred = final_model.predict(X_test)
  
  return y_pred