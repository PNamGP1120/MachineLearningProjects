import numpy as np
import pandas as pd

# Thiết lập random seed để có dữ liệu giống nhau mỗi lần chạy
np.random.seed(42)

# Tạo 1000 bệnh nhân với các chỉ số ngẫu nhiên
num_samples = 1000

# Tuổi từ 20 đến 80
age = np.random.randint(20, 80, num_samples)

# BMI từ 18.5 đến 35 (phù hợp với chỉ số cơ thể bình thường - béo phì)
bmi = np.random.uniform(18.5, 35, num_samples)

# Lượng đường trong máu từ 70 đến 200 mg/dL
glucose = np.random.uniform(70, 200, num_samples)

# Xác định bệnh tiểu đường (1 nếu glucose > 126 và BMI cao, ngược lại là 0)
diabetes = ((glucose > 126) & (bmi > 25)).astype(int)

# Tạo DataFrame
df = pd.DataFrame({'Age': age, 'BMI': bmi, 'Glucose': glucose, 'Diabetes': diabetes})

# Lưu bộ dữ liệu vào file CSV
df.to_csv("diabetes_dataset.csv", index=False)

print("Bộ dữ liệu đã được tạo và lưu vào file 'diabetes_dataset.csv'")
