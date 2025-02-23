import pandas as pd

df = pd.read_csv('IRIS.csv')

# print(df.head())
#
# print(df['sepal_length'])
#
# print(df[df['sepal_length']>5])

# print(df[['sepal_length', 'sepal_width']])

print(df.iloc[0])

print(df.iloc[0, 1])

print(df.loc[0, "petal_length"])




