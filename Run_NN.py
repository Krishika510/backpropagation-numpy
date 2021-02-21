import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import fncs as nn

from sklearn.datasets import fetch_openml
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
mnist = fetch_openml('mnist_784',version=1)

# Getting training and testing data:
# We are setting up just a simple binary classification problem in which we aim to
# properly classify the number 2.
X, y_str = mnist["data"], mnist["target"]
y = np.array([int(int(i)==2) for i in y_str])
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

# Creating a neural network structure
net = nn.net_create([784,100,1])

# Training neural network:
# Note that since I am not doing any hyper-parameter tuning, I am using the test sets for
# validation to show how the generalization error changes as the network gets trained.
Loss,Loss_val,mae_val = nn.net_train(net,X_train,y_train,X_test,y_test,epsilon=1e-6,NIter=300)

# Plotting learning curves:
# Note that we don't observe overfitting here because the model is very simple.
plt.plot(Loss/np.max(Loss))
plt.plot(Loss_val/np.max(Loss_val))
plt.legend({'Normalized Training Loss','Normalized Validation Loss'})
plt.show()