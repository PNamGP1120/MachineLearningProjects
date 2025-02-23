import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('IRIS.csv')

fig, ax = plt.subplots()

# Tạo danh sách các loài hoa và màu sắc tương ứng
species = df['species'].unique()
colors = plt.cm.Paired(range(len(species)))
species_color_dict = dict(zip(species, colors))

# Vẽ các điểm scatter dưới dạng tọa độ phức
for datapoint in df.index:
    species = df.loc[datapoint, 'species']

    # Chuyển sepal_length và sepal_width thành phần thực và ảo của số phức
    real_part = df.loc[datapoint, 'sepal_length']  # Phần thực
    imaginary_part = df.loc[datapoint, 'sepal_width']  # Phần ảo

    # Biểu diễn điểm dưới dạng số phức
    complex_point = real_part + 1j * imaginary_part  # a + bi

    # Vẽ điểm scatter với tọa độ phức
    ax.scatter(complex_point.real, complex_point.imag, color=species_color_dict[species], marker='o')

# Thêm nhãn và hiển thị
ax.set_xlabel('Real Part (Sepal Length)')
ax.set_ylabel('Imaginary Part (Sepal Width)')

plt.show()
