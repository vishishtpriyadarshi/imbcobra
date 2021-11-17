from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier


def logistic_regression(X_train, y_train, X_test):
  print("[Executing]: Running Logistic Regression model ...\n")

  # Model Fitting
  model = LogisticRegression(random_state = 21)
  model.fit(X_train, y_train)

  # Predictions on test dataset
  y_pred = model.predict(X_test)
  
  return y_pred


def logistic_regression_adam(X_train, y_train, X_test):
  print("[Executing]: Running Logistic Regression model with Adam ...\n")

  # Model Fitting
  model = MLPClassifier(hidden_layer_sizes=(1), solver='adam', random_state=48)
  model.fit(X_train, y_train)

  # Predictions on test dataset
  y_pred = model.predict(X_test)
  
  return y_pred