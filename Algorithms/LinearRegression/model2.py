import numpy as np
from matplotlib import pyplot as plt


class LinearRegression:
    def __init__(self, learning_rate=0.000001, tolerance = 1e-6):
        self.weights = None
        self.learning_rate = learning_rate
        self.tolerance = tolerance
        self.loss_history = []

    def forward_propagation(self, X):
        """Tính dự đoán: y_predict = XTxm"""
        y_predict = np.dot(X.T, self.weights)
        return y_predict

    def cost_function(self, predictions, y):
        """Hàm loss (Mean Squared Error - MSE):
                   L = (1/n) * sum((y - y_predict)^2)
        """
        loss = np.mean((y - predictions) ** 2)
        return loss

    def backward_propagation(self, X):
        """Tính gradient của XTm"""

        return X

    def update_parameters(self, X):
        """Cập nhật tham số m, c theo gradient descent"""
        self.weights -= self.learning_rate * self.backward_propagation(X)

    def train(self, X, y, iterations=2000):

        self.weights = np.random.uniform(-1, 1, (X.shape[1], 1))  # Tạo ma trận (n,1)



        for i in range(iterations):
            predictions = self.forward_propagation(X)
            cost = self.cost_function(predictions, y)

            # Kiểm tra hội tụ
            if i > 0 and abs(self.loss_history[-1] - cost) < self.tolerance:
                print(f"Gradient Descent hội tụ tại vòng {i + 1}")
                break

            x = self.backward_propagation(X)

            self.update_parameters(x)
            self.loss_history.append(cost)

        # print(f"Huấn luyện hoàn tất: m = {self.m:.4f}, c = {self.c:.4f}")

    def predict(self, X):
        """Dự đoán đầu ra với đầu vào X"""
        return self.forward_propagation(X)

    def evaluate(self, X, y):
        """Đánh giá mô hình trên tập kiểm tra"""
        predictions = self.predict(X)

        return self.cost_function(predictions, y)

    def plot_regression_line(self, X, y):
        """
        Vẽ biểu đồ tuyến tính cho mỗi feature.
        Với dữ liệu nhiều chiều, ta vẽ mối quan hệ của từng feature với y
        khi các feature khác được giữ cố định ở giá trị trung bình.
        """
        for i in range(X.shape[1]):  # Lặp qua từng feature
            plt.figure()
            plt.scatter(X[:, i], y, color='blue', label=f"Actual Data (Feature {i + 1})")

            # Giữ các feature khác ở giá trị trung bình
            mean_other_features = np.mean(np.delete(X, i, axis=1), axis=0)
            x_vals = np.linspace(X[:, i].min(), X[:, i].max(), 100)
            # Tạo ma trận X_temp: các cột khác được giữ cố định ở trung bình,
            # cột thứ i thay bằng các giá trị x_vals
            X_temp = np.tile(mean_other_features, (len(x_vals), 1))
            X_temp = np.insert(X_temp, i, x_vals, axis=1)

            y_vals = np.dot(X_temp, self.m) + self.c
            plt.plot(x_vals, y_vals, color='red', label=f"Predicted (Feature {i + 1})")
            plt.xlabel(f"X - Feature {i + 1}")
            plt.ylabel("Y - Output")
            plt.title(f"Linear Regression (Feature {i + 1})")
            plt.legend()
            plt.show()