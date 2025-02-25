import numpy as np

import pandas as pd
from matplotlib import pyplot as plt

df = pd.read_csv("data_for_lr.csv")


train_input = np.array(df[:int(len(df) * 0.8)][['x']]).reshape(-1, 1)
train_output = np.array(df[:int(len(df) * 0.8)]['y']).reshape(-1, 1)
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


plt.figure(figsize=(16, 12))
plt.scatter(df['x'], df['y'])
plt.plot(df['x'], [lr.w[0]+lr.w[1]*i for i in df['x']], color='red')
print(lr.w)
plt.show()