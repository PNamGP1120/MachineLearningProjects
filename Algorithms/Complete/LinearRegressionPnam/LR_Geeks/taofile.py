import numpy as np
import pandas as pd

# Số lượng mẫu
N = 100  # 100 điểm dữ liệu

# Tham số đường thẳng y = ax + b
a = 3.5  # Hệ số góc
b = 2.0  # Hệ số chặn

# Sinh dữ liệu đầu vào x
np.random.seed(42)  # Để kết quả giống nhau mỗi lần chạy
x = np.random.uniform(-10, 10, size=(N, 1))  # Giá trị x từ -10 đến 10

# Sinh nhiễu (noise) từ phân phối chuẩn
noise = np.random.normal(0, 2, size=(N, 1))  # Nhiễu có độ lệch chuẩn 2

# Tạo đầu ra y theo công thức y = ax + b + noise
y = a * x + b + noise

# Lưu dữ liệu vào file CSV
df = pd.DataFrame({'x': x.flatten(), 'y': y.flatten()})
df.to_csv("data_for_lr2.csv", index=False)

print("Dữ liệu đã được tạo và lưu vào 'data_for_lr.csv'.")

