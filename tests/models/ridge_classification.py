from sklearn.linear_model import RidgeClassifier

def ridge_classification(X_train, y_train, X_test):
  print("[Executing]: Running Ridge Regression model ...\n")

  # Model Fitting
  model = RidgeClassifier(random_state=88)
  model.fit(X_train, y_train)

  # Predictions on test dataset
  y_pred = model.predict(X_test)
  
  return y_pred