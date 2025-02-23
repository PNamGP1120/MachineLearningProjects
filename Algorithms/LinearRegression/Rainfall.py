import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



from model import LinearRegression


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

df = pd.read_csv("../Complete/LinearRegressionPnam/RainfallProject/Austin-2019-01-01-to-2023-07-22.csv")
train_input = np.array(df[:int(len(df) * 0.8)][['tempmax']])
train_output = np.array(df[:int(len(df) * 0.8)]['precip'])

test_input = np.array(df[int(len(df) * 0.8):][['tempmax']])
test_output = np.array(df[int(len(df) * 0.8):]['precip'])

lr = LinearRegression()
lr.train(train_input, train_output)
lr.plot_regression_line(train_input, train_output)

print(lr.predict(test_input))
print(test_output)