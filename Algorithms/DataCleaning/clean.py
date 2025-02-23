import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('Titanic-Dataset.csv')
print(df[df.duplicated()])

df_duplicated_all = df[df.duplicated(keep=False)]
print(df_duplicated_all)

plt.figure(figsize=(16, 8))
# plt.scatter(df['PassengerId'], df['Survived'])
# plt.scatter(df['PassengerId'], df['Pclass'])

for i in range(len(df['Age'])):

    if np.isnan(df['Age'][i]):
        plt.scatter(df['PassengerId'][i],-1, color='red')
    else:
        plt.scatter(df['PassengerId'][i],  df['Age'][i], color='blue')

plt.show()
