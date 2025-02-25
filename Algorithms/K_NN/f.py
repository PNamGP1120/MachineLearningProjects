import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets

# Load dữ liệu Iris
iris = datasets.load_iris()
X = iris.data[:, :2]  # Chỉ lấy 2 đặc trưng đầu tiên (Sepal Length, Sepal Width)
y = iris.target        # Nhãn (0: setosa, 1: versicolor, 2: virginica)

# Chuyển thành DataFrame
df = pd.DataFrame(X, columns=['Sepal Length', 'Sepal Width'])
df['species'] = np.array(['setosa', 'versicolor', 'virginica'])[y]  # Gán tên lớp

# Lưu thành CSV
df.to_csv('iris_2features.csv', index=False)

# Vẽ biểu đồ phân tán của dữ liệu
plt.figure(figsize=(8, 6))
for i, species in enumerate(['setosa', 'versicolor', 'virginica']):
    plt.scatter(df[df['species'] == species]['Sepal Length'],
                df[df['species'] == species]['Sepal Width'],
                label=species)

plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.title('Iris Dataset (2 Features)')
plt.legend()
plt.show()
