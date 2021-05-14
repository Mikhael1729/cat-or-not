import numpy as np
import copy
from lr_utils import load_dataset

def sigmoid(z):
  """
  Returns the sigmoid value of the given z scalar or numpy matrix.
  """
  result = 1 / (1 + np.exp(-z))
  return result

def initialize_with_zeros(input_size):
  """
  Initialize weights of the model with shape (input_size, 1) and the bias to 0.

  Arguments:
  input_size -- quantity of neurons in the input layer

  Returns:
  w -- a matrix of shape (dim, 1) initilized with 0.
  b -- the bias a scalar number initialized at 0.
  """
  w = np.zeros((input_size, 1))
  b = 0

  return w, b

def propagate(w, b, X, Y):
  """
  Compute the values for the forward and back processs.

  Variables:
  input_size -- The quantity of neurons in the input layer
  batch_size -- The number of sampples on a given batch

  Arguments:
  w -- weights, a numpy matrix of shape (input_size * input_size * 3, 1)
  b -- bias, a scalar
  X -- input values, a numpy matrix of shape (input_size * input_size * 3, batch_size)
  Y -- true labels, a numpy matrix of shape (1, batch_size), containing 0 if non-cat, 1 if cat

  Return:
  cost -- Negative log-likehood cost for logistic regression
  dw -- gradient of the loss with respect to w, thus same shape as w
  db -- gradient of the loss with respect to b, thus same shape as b
  """
  m = X.shape[1] # Number of samples

  # Forward propagation: computes A and the cost.
  A = sigmoid(np.dot(w.T, X) + b)
  cost = -(1 / m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))

  # Backward propagation: find gradients.
  dw = (1 / m) * np.dot(X, (A - Y).T)
  db = (1 / m) * np.sum(A - Y)

  cost = np.squeeze(np.array(cost))
  gradients = { "dw": dw, "db": db }

  return gradients, cost

def train(w, b, X, Y, num_iterations=100, learning_rate=0.009, print_cost=False):
  w = copy.deepcopy(w)
  b = copy.deepcopy(b)

  costs = []
  for i in range(num_iterations):
    grads, cost = propagate(w, b, X, Y)

    dw = grads["dw"]
    db = grads["db"]
    w = w - learning_rate * dw
    b = b - learning_rate * db

    # Store and print the cost each 100 iterations
    if i % 100 == 0:
      costs.append(cost)
      if print_cost:
        print(f"Iteration {i} --> cost = {cost}")

  params = { "w": w, "b": b }
  grads = { "dw": dw, "db": db }

  return params, grads, costs

def predict(w, b, X):
  m = X.shape[1]
  Y_prediction = np.zeros((1, m))
  w = w.reshape(X.shape[0], 1)

  # Forward
  A = sigmoid(np.dot(w.T, X) + b)

  for i in range(A.shape[1]):
    if A[0, i] > 0.5:
      Y_prediction[0, i] = 1
    else:
      Y_prediction[0, i] = 0

  return Y_prediction

def compute_model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=False):
  w, b = initialize_with_zeros(X_train.shape[0])

  params, grads, costs = train(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)
  w = params["w"]
  b = params["b"]

  Y_prediction_test = predict(w, b, X_test)
  Y_prediction_train = predict(w, b, X_train)

  if print_cost:
    train_accuracy = 100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100
    test_accuracy = 100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100
    print(f"train accuracy: {train_accuracy}")
    print(f"test accuracy: {test_accuracy}")

  d = {
    "costs": costs,
    "Y_prediction_test": Y_prediction_test, 
    "Y_prediction_train" : Y_prediction_train, 
    "w" : w, 
    "b" : b,
    "learning_rate" : learning_rate,
    "num_iterations": num_iterations
  }

if __name__ == "__main__":
  # Load the data
  train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

  train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
  test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

  # Standarize data set.
  train_set_x = train_set_x_flatten / 255.
  test_set_x = test_set_x_flatten / 255.

  logistic_regression_model = compute_model(
    train_set_x,
    train_set_y,
    test_set_x,
    test_set_y,
    num_iterations=2000,
    learning_rate=0.005,
    print_cost=True
  )
