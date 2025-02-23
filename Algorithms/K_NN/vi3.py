import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('IRIS.csv')

fig, ax = plt.subplots()

species = df['species'].unique()
colors = plt.cm.Paired(range(len(species)))
species_color_dict = dict(zip(species, colors))
# print(species_color_dict)
# for index, specie in enumerate(species):
#     print(index, specie, colors[index])
#     ax.plot(df.index,df['sepal_length'], colors[index])
#
#
for datapoint in df.index:
    ax.scatter(datapoint, df.loc[datapoint, 'sepal_length'], color = species_color_dict[df.loc[datapoint, 'species']], marker = 'o')
    ax.scatter(datapoint, df.loc[datapoint, 'sepal_width'], color=species_color_dict[df.loc[datapoint, 'species']], marker='.')
    ax.scatter(datapoint, df.loc[datapoint, 'petal_length'], color=species_color_dict[df.loc[datapoint, 'species']], marker='v')
    ax.scatter(datapoint, df.loc[datapoint, 'petal_width'], color=species_color_dict[df.loc[datapoint, 'species']], marker='+')


ax.plot(df.index, df['sepal_length'], color='red', linestyle='-', alpha=0.5, label='Sepal Length')
ax.plot(df.index, df['sepal_width'], color='blue', linestyle='-', alpha=0.5, label='Sepal Width')
ax.plot(df.index, df['petal_length'], color='green', linestyle='-', alpha=0.5, label='Petal Length')
ax.plot(df.index, df['petal_width'], color='purple', linestyle='-', alpha=0.5, label='Petal Width')

ax.set_xlabel('Index')
ax.set_ylabel('Values')
ax.legend()

plt.show()