import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('diabetes_dataset.csv')
# Vẽ biểu đồ phân tán
plt.scatter(df['Glucose'], df['BMI'], c=df['Diabetes'], cmap='bwr', alpha=0.6)

plt.xlabel("Lượng đường trong máu (Glucose Level)")
plt.ylabel("Chỉ số BMI")
plt.title("Phân loại bệnh tiểu đường dựa trên Glucose & BMI")
plt.colorbar(label="Diabetes (0 = Không, 1 = Có)")
plt.show()
