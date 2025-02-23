import pandas as pd
import matplotlib.pyplot as plt

# Giả sử dữ liệu của bạn đã có trong DataFrame df
df = pd.read_csv('IRIS.csv')

# Tạo một cột index
df['index'] = df.index

# Tạo một subplot với 4 biểu đồ con
fig, ax = plt.subplots()

# Vẽ các dòng cho 4 giá trị của iris
colors = {'setosa': 'red', 'versicolor': 'green', 'virginica': 'blue'}

for species in df['species'].unique():
    subset = df[df['species'] == species]
    for column in ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']:
        ax.plot(subset['index'], subset[column], color=colors[species], label=f'{species} - {column}')

# Tô màu cho từng species
ax.legend()
plt.xlabel('Index')
plt.ylabel('Values')
plt.title('Visualization of Iris Dataset')
plt.show()
