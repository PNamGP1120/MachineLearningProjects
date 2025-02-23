import numpy as np
import pandas as pd

# Tạo dữ liệu ngẫu nhiên
np.random.seed(42)
num_samples = 100

# Lượng đường trong máu từ 70 đến 200 mg/dL
glucose = np.random.uniform(70, 200, num_samples)

# Quy tắc: Nếu glucose > 126 thì có nguy cơ mắc bệnh (1), ngược lại là không (0)
diabetes = (glucose > 126).astype(int)

# Tạo DataFrame
df = pd.DataFrame({'Glucose': glucose, 'Diabetes': diabetes})

# Xuất dữ liệu ra file CSV
df.to_csv("glucose_diabetes.csv", index=False)

print("Bộ dữ liệu đã được tạo và lưu vào file 'glucose_diabetes.csv'")
