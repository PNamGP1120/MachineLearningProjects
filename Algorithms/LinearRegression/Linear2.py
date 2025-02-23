import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes as ax
from matplotlib.animation import FuncAnimation

data = pd.read_csv('../Complete/LinearRegressionPnam/LR_Geeks/data_for_lr.csv')


# Drop the missing values
data = data.dropna()

# training dataset and labels
train_input = np.array(data.x[0:50]).reshape(50, 1)
train_output = np.array(data.y[0:50]).reshape(50, 1)

# valid dataset and labels
test_input = np.array(data.x[500:700]).reshape(199, 1)
test_output = np.array(data.y[500:700]).reshape(199, 1)


class LinearRegression:
    def __init__(self, learning_rate=0.0001, tolerance = 1e-6):
        self.m = np.random.uniform(-1, 1)
        self.c = np.random.uniform(-1, 1)
        self.learning_rate = learning_rate
        self.tolerance = tolerance
        self.loss_history = []

    def forward_propagation(self, X):
        """Phương trình y_predict = mx + c"""
        return self.m * X + self.c

    def cost_function(self, predictions, y):
        """Loss function (Mean Squared Error - MSE) L = (y-y_pre)^2/n"""
        return np.mean((y - predictions) ** 2)

    def backward_propagation(self, X, y, predictions):
        """Tính gradient của m và c"""
        errors = predictions - y
        dm = 2 * np.mean(errors * X)
        dc = 2 * np.mean(errors)
        return dm, dc

    def update_parameters(self, dm, dc):
        """Cập nhật tham số m, c theo gradient descent"""
        self.m -= self.learning_rate * dm
        self.c -= self.learning_rate * dc

    def train(self, X, y, iterations=2000):
        """Huấn luyện mô hình với số vòng lặp cụ thể"""
        for i in range(iterations):
            predictions = self.forward_propagation(X)
            cost = self.cost_function(predictions, y)

            # Kiểm tra hội tụ
            if i > 0 and abs(self.loss_history[-1] - cost) < self.tolerance:
                print(f"Gradient Descent hội tụ tại vòng {i + 1}")
                break

            dm, dc = self.backward_propagation(X, y, predictions)
            self.update_parameters(dm, dc)
            self.loss_history.append(cost)

        print(f"Huấn luyện hoàn tất: m = {self.m:.4f}, c = {self.c:.4f}")

    def predict(self, X):
        """Dự đoán đầu ra với đầu vào X"""
        return self.forward_propagation(X)

    def evaluate(self, X, y):
        """Đánh giá mô hình trên tập kiểm tra"""
        predictions = self.predict(X)
        return self.cost_function(predictions, y)


    def plot_regression_line(self, X, y):
        """Vẽ dữ liệu thực tế và đường hồi quy"""
        plt.scatter(X, y, color='blue', label="Dữ liệu thực tế")
        x_vals = np.linspace(X.min(), X.max(), 100)
        y_vals = self.m * x_vals + self.c
        plt.plot(x_vals, y_vals, color='red', label=f"y = {self.m:.2f}x + {self.c:.2f}")
        plt.xlabel("X - Đầu vào")
        plt.ylabel("Y - Đầu ra")
        plt.title("Hồi quy tuyến tính")
        plt.legend()
        plt.show()

model = LinearRegression(learning_rate=0.0001, tolerance=1e-6)
model.train(train_input, train_output)

print(model.evaluate(test_input, test_output))
model.plot_regression_line(test_input, test_output)

print(train_input.shape, train_output.shape)