import numpy as np

import pandas as pd
from matplotlib import pyplot as plt

df = pd.read_csv("Austin-2019-01-01-to-2023-07-22.csv")


train_input = np.array(df[:int(len(df) * 0.8)][['tempmax', 'tempmin', 'dew', 'humidity']])
train_output = np.array(df[:int(len(df) * 0.8)]['precip']).reshape(-1, 1)
train_input = np.hstack((np.ones((train_input.shape[0], 1)), train_input))




class LinearRegression:
    def __init__(self):
        self.w = None

    def fit(self, X, y):
        XTX = np.linalg.pinv(X.T @ X)
        XTy = X.T @ y
        self.w = XTX @ XTy

    def predict(self, X):
        return X @ self.w



lr = LinearRegression()
lr.fit(train_input, train_output)


print(lr.w)