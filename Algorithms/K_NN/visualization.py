import numpy as np
import pandas
import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv('IRIS.csv')

species_unique = data['species'].unique()
colors_map = dict(zip(species_unique, ['lightcoral', 'lightgreen', 'lightblue']))


fig, ax = plt.subplots(figsize=(20,12))





ax.plot(data.index, data['sepal_length'], 'r', label='Sepal Length')
ax.plot(data.index, data['sepal_width'], 'b', label='Sepal Width')
ax.plot(data.index, data['petal_length'], 'g', label='Petal Length')
ax.plot(data.index, data['petal_width'], 'y', label='Petal Width')
ax.set_title('Fill Region for All Values by Species')
ax.set_xlabel('Index')
ax.set_ylabel('Length / Width')
ax.legend()

plt.show()