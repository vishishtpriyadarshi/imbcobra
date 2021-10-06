from sklearn.linear_model import LogisticRegression

def logistic_regression(X_train, y_train, X_test):
  print("[Executing]: Running Logistic Regression model ...\n")

  # Model Fitting
  model = LogisticRegression(random_state = 0, solver='lbfgs', multi_class='multinomial')
  model.fit(X_train, y_train)

  # Predictions on test dataset
  y_pred = model.predict(X_test)
  
  return y_pred