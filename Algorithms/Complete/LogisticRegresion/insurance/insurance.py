import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from model import LogisticRegression

df = pd.read_csv("insurance_data.csv")
df.sort_values('bought_insurance')

X_train = df["age"].values.reshape(-1, 1)  # Biến đầu vào (Tuổi)
y_train = df["bought_insurance"].values

model = LogisticRegression(X_train, y_train)

model.train(lamda=0.001, lr=0.0001, nepochs=10000)

y_pred = model.predict(X_train)
print("Dự đoán:", y_pred)
print("Độ chính xác:", np.mean(y_pred == y_train) * 100, "%")

model.plot_decision_boundary()
