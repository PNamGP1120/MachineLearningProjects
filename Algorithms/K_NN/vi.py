import pandas as pd
import matplotlib.pyplot as plt

# Đọc dữ liệu Iris từ file CSV
data = pd.read_csv('IRIS.csv')

# Khởi tạo đồ thị và kích thước
fig, ax = plt.subplots(figsize=(15, 8))

# Màu sắc cho từng loài cần tô màu
colors_map = {
    "Iris-setosa": "lightblue",
    "Iris-versicolor": "lightgreen",
    "Iris-virginica": "lightcoral"
}

# Chọn 3 loài cần phân loại
selected_species = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]

# Vẽ các vùng và dữ liệu theo từng loài trong selected_species
for species, subset in data.groupby('species'):
    if species in selected_species:
        ax.plot(subset.index, subset['sepal_length'], label=f'{species} - Sepal Length', marker='o')
        ax.plot(subset.index, subset['sepal_width'], label=f'{species} - Sepal Width', linestyle='--')
        ax.plot(subset.index, subset['petal_length'], label=f'{species} - Petal Length', marker='x')
        ax.plot(subset.index, subset['petal_width'], label=f'{species} - Petal Width', linestyle='dotted')

        # Tô màu vùng dữ liệu theo loài
        ax.fill_between(subset.index, subset['sepal_length'], subset['petal_length'], color=colors_map[species],
                        alpha=0.2)

# Đặt tiêu đề và ghi chú
ax.set_title('Trực quan hóa dữ liệu Iris với vùng tô theo 3 loài')
ax.set_xlabel('Index')
ax.set_ylabel('Chiều dài / chiều rộng (cm)')
ax.legend()
plt.show()
