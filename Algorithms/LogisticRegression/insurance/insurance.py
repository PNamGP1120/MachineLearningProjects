import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from model import LogisticRegression


# # Tạo dữ liệu đơn giản với 1 feature
# X_train = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
# y_train = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])  # Nhãn nhị phân


df = pd.read_csv("insurance_data.csv")
df.sort_values('bought_insurance')
# Chuyển đổi dữ liệu
X_train = df["age"].values.reshape(-1, 1)  # Biến đầu vào (Tuổi)
y_train = df["bought_insurance"].values
print(X_train, y_train)
# Khởi tạo model
model = LogisticRegression(X_train, y_train)

# Train model
model.train(lamda=0.001, lr=0.0001, nepochs=10000)

# Dự đoán trên tập huấn luyện
y_pred = model.predict(X_train)
print("Dự đoán:", y_pred)
print("Độ chính xác:", np.mean(y_pred == y_train) * 100, "%")

model.plot_decision_boundary()
