import pandas as pd
import numpy as np
# Input data
data = pd.read_csv('https://bit.ly/2KF29Bd', header=0)
X = data.iloc[:, 0].values
y = data.iloc[:, 1].values
n = data.shape[0] # rows
# Building the model
m = 0.0
b = 0.0
sample_size = 1 # sample size, inrease the sample_size to more than 1, this becomes SGD
L = .0001 # The learning Rate
epochs = 1_000_000 # The number of iterations to perform gradient descent
# Performing Stochastic Gradient Descent
for i in range(epochs):
    idx = np.random.choice(n, sample_size, replace=False)
    x_sample = X[idx]
    y_sample = y[idx]
    # The current predicted value of y
    y_pred = m * x_sample + b
    # d/dm derivative of loss function
    D_m = (-2 / sample_size) * sum(x_sample * (y_sample - y_pred))
    # d/db derivative of loss function
    D_b = (-2 / sample_size) * sum(y_sample - y_pred)
    m = m - L * D_m # Update m
    b = b - L * D_b # Update b
    # print progress
    if i % 10000 == 0:
        print(i, m, b)
print("y = {0}x + {1}".format(m, b))