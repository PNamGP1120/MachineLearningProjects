import numpy as np

from Algorithms.NeuralNetwork.model import MLPClassifier, ThreeSpiralDataset

import matplotlib.pyplot as plt

# Tạo dữ liệu Three Spirals
dataset = ThreeSpiralDataset(n_points=200, n_classes=3, noise=0.2)
X, y = dataset.X, dataset.y

# Chia dữ liệu thành tập train và test mà không dùng sklearn
N = X.shape[0]
indices = np.random.permutation(N)
split = int(0.8 * N)  # 80% train, 20% test

X_train, y_train = X[indices[:split]], y[indices[:split]]
X_test, y_test = X[indices[split:]], y[indices[split:]]

# Khởi tạo và huấn luyện mô hình MLP
d0, d1, d2 = 2, 100, 3  # Input layer (2), Hidden layer (100), Output layer (3)
mlp = MLPClassifier(d0, d1, d2, eta=1)
mlp.train(X_train, y_train, epochs=5000)

# Dự đoán trên tập train và test
y_pred_train = mlp.predict(X_train)
y_pred_test = mlp.predict(X_test)

# Đánh giá độ chính xác không dùng sklearn
train_acc = np.mean(y_pred_train == y_train) * 100
test_acc = np.mean(y_pred_test == y_test) * 100

print(f'Training Accuracy: {train_acc:.2f}%')
print(f'Test Accuracy: {test_acc:.2f}%')


# Vẽ biểu đồ kết quả phân loại với decision boundary
plt.figure(figsize=(6, 6))

# Tạo lưới điểm để vẽ boundary
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
Z = mlp.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Vẽ boundary
plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred_test, cmap='coolwarm', edgecolors='k', alpha=0.7)
plt.title("MLP Classification with Decision Boundary")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()



