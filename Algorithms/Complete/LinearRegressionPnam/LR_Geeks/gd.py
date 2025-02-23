import numpy as np

import pandas as pd
from matplotlib import pyplot as plt

df = pd.read_csv("data_for_lr.csv")


train_input = np.array(df[:int(len(df) * 0.8)][['x']])
train_output = np.array(df[:int(len(df) * 0.8)]['y']).reshape(-1, 1)
train_input = np.hstack((np.ones((train_input.shape[0], 1)), train_input))




class LinearRegressionGD:
    def __init__(self, learning_rate = 0.00001, iters = 1000):
        self.w = None
        self.learning_rate = learning_rate
        self.iters = iters

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros((n_features, 1))  # Khởi tạo trọng số w = 0

        for i in range(self.iters):
            gradient = (1 / n_samples) * (X.T @ (X @ self.w - y))  # Tính gradient
            self.w -= self.learning_rate * gradient  # Cập nhật w
            loss = (1 / (2 * n_samples)) * np.sum((X @ self.w - y) ** 2)
            print(f"Epoch {i}: Loss = {loss:.4f}")
            # Tính loss sau mỗi 100 epoch để theo dõi


    def predict(self, X):
        return X @ self.w

lr = LinearRegressionGD(learning_rate=0.0001, iters=90)

lr.fit(train_input, train_output)


plt. figure(figsize=(16, 12))
plt.scatter(df['x'], df['y'])
plt.plot(df['x'], [lr.w[0]+lr.w[1]*i for i in df['x']], color='red')
print(lr.w)
plt.show()