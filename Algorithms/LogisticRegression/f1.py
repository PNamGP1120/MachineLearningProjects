import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing

# Tải dữ liệu
data = fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['Target'] = data.target  # Thêm cột giá nhà

# Hiển thị mối quan hệ giữa các đặc trưng
sns.pairplot(df[['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'Target']])
plt.show()


plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Heatmap: Mức độ tương quan giữa các biến")
plt.show()

plt.figure(figsize=(8, 5))
sns.histplot(df['Target'], bins=30, kde=True)
plt.xlabel("Giá nhà (100,000$)")
plt.ylabel("Tần suất")
plt.title("Biểu đồ phân bố giá nhà")
plt.show()
