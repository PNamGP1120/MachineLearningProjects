import numpy as np
from sklearn.linear_model import LinearRegression

import pandas as pd

df = pd.read_csv('data_for_lr.csv')
train_input = np.array(df[:int(len(df) * 0.8)][['x']]).reshape(-1, 1)
train_output = np.array(df[:int(len(df) * 0.8)]['y']).reshape(-1, 1)
train_input = np.hstack((np.ones((train_input.shape[0], 1)), train_input))
lr = LinearRegression()

lr.fit(train_input, train_output)

print(lr.coef_)