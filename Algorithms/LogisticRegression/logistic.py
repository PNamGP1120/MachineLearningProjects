import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from model import LogisticRegression

# Hiển thị tất cả các cột, các dòng
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

df = pd.read_csv('framingham.csv')
# print(df.head())
# print(df.shape)

train_input = np.array(df[:int(len(df) * 0.8)].drop(columns='TenYearCHD'))
train_output = np.array(df[:int(len(df) * 0.8)]['TenYearCHD'])

test_input = np.array(df[int(len(df) * 0.8):].drop(columns='TenYearCHD'))
test_output = np.array(df[int(len(df) * 0.8):]['TenYearCHD'])


model = LogisticRegression()
model = model.train(train_input, train_output, 20)
print(model.m, model.c)


