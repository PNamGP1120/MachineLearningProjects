import numpy as np

import pandas as pd
from matplotlib import pyplot as plt

from model import LinearRegression
df = pd.read_csv("data_for_lr.csv")

train_input = np.array(df[:int(len(df) * 0.8)][['x']])
train_output = np.array(df[:int(len(df) * 0.8)]['y'])
plt.scatter(train_input, train_output)
test_input = np.array(df[int(len(df) * 0.8):][['x']])
test_output = np.array(df[int(len(df) * 0.8):]['y'])

lr = LinearRegression()
lr.train(train_input, train_output)
plt.scatter(train_input, [lr.m*i+lr.c for i in train_input], color='red')
plt.show()
