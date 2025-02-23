# %%
import matplotlib.pyplot as plt

import pandas as pd

df = pd.read_csv('glucose_diabetes.csv')
plt.figure(figsize=(16,12))
# %%

# %%
plt.scatter(df['Glucose'], df['Diabetes'], c=df['Diabetes'], cmap='bwr', alpha=0.7)
plt.xlabel("Lượng đường trong máu (Glucose Level)")
plt.ylabel("Bệnh tiểu đường (0 = Không, 1 = Có)")
plt.title("Phân loại bệnh tiểu đường dựa trên chỉ số Glucose")
# %%
# %%
plt.show()

# %%