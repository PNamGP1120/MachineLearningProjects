import numpy as np
import matplotlib.pyplot as plt

class LogisticRegression:
    def __init__(self, X, y):
        """
        Khởi tạo mô hình Logistic Regression với SGD.
        - X: Ma trận đầu vào (N, d) (chưa có bias)
        - y: Vector nhãn (N,)
        """
        self.X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)  # Thêm cột bias
        self.y = y
        self.weights = np.random.randn(self.X.shape[1])*0.01  # Khởi tạo ngẫu nhiên
        self.loss_hist = []

    def sigmoid(self, z):
        """Tính sigmoid của z."""
        return 1 / (1 + np.exp(-z))

    def prob(self):
        """Tính xác suất P(y=1|X)."""
        return self.sigmoid(self.X @ self.weights)

    def loss(self, lamda):
        """Tính Binary Cross-Entropy Loss có Regularization"""
        p = self.prob()
        eps = 1e-8  # Tránh log(0)
        return np.mean(-self.y * np.log(p + eps) - (1 - self.y) * np.log(1 - p + eps)) + lamda * np.sum(
            self.weights ** 2) / 2

    def train(self, lamda=0.001, lr=0.01, nepochs=100):
        """Huấn luyện mô hình bằng Stochastic Gradient Descent (SGD)."""
        N, d = self.X.shape
        self.loss_hist = [self.loss(lamda)]  # Lưu lịch sử loss

        for ep in range(nepochs):
            mix_ids = np.random.permutation(N)  # Shuffle dữ liệu
            for i in mix_ids:
                xi = self.X[i]  # Lấy một mẫu
                yi = self.y[i]
                zi = self.sigmoid(np.dot(xi, self.weights))  # Dự đoán xác suất
                self.weights -= lr * ((zi - yi) * xi + lamda * self.weights)  # Cập nhật SGD

            self.loss_hist.append(self.loss(lamda))  # Lưu loss mỗi epoch

            # Điều kiện dừng sớm nếu trọng số không thay đổi nhiều
            if len(self.loss_hist) > 1 and abs(self.loss_hist[-1] - self.loss_hist[-2]) < 1e-15:
                break

        return self.weights

    def predict(self, X_test):
        """Dự đoán nhãn (0 hoặc 1) cho dữ liệu mới."""
        X_test = np.concatenate((np.ones((X_test.shape[0], 1)), X_test), axis=1)  # Thêm bias
        return (self.sigmoid(X_test @ self.weights) >= 0.5).astype(int)  # Ngưỡng 0.5

    def plot_decision_boundary(self):
        """Vẽ Decision Boundary (chỉ hỗ trợ cho 1D feature)."""
        if self.X.shape[1] > 2:
            print("Chỉ hỗ trợ vẽ cho dữ liệu có 1 feature!")
            return

        # Tạo lưới điểm để vẽ bề mặt dự đoán
        x_min, x_max = self.X[:, 1].min() - 1, self.X[:, 1].max() + 1
        x_values = np.linspace(x_min, x_max, 100).reshape(-1, 1)

        # Tạo bias và dự đoán xác suất
        X_bias = np.concatenate((np.ones((x_values.shape[0], 1)), x_values), axis=1)
        y_probs = self.sigmoid(X_bias @ self.weights)

        # Vẽ dữ liệu thật
        plt.scatter(self.X[:, 1], self.y, c=self.y, cmap="coolwarm", edgecolors="k", label="Dữ liệu thực tế")

        # Vẽ đường quyết định
        plt.plot(x_values, y_probs, label="Decision Boundary (Sigmoid)", color="black")

        # Đánh dấu ngưỡng 0.5
        plt.axhline(0.5, linestyle="dashed", color="gray", label="Ngưỡng 0.5")

        plt.xlabel("Feature")
        plt.ylabel("Xác suất P(y=1|X)")
        plt.title("Decision Boundary của Logistic Regression")
        plt.legend()
        plt.show()
