import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv("../Complete/LinearRegressionPnam/RainfallProject/Austin-2019-01-01-to-2023-07-22.csv")
df =df.dropna()
X = np.array(df[['tempmax', 'tempmin', 'dew', 'humidity']])


# Chọn 4 đặc trưng để huấn luyện (giả sử có các cột x1, x2, x3, x4)
train_input = np.array(df[['tempmax', 'tempmin', 'dew', 'humidity']][0:50])  # 4 đặc trưng cho 50 mẫu
train_output = np.array(df['tempmax'][0:50]).reshape(50, 1)




class LinearRegression:
    def __init__(self, learning_rate=0.0001, tolerance=1e-6):
        # Khởi tạo trọng số và hệ số (cho 4 đặc trưng)
        self.weights = np.random.uniform(-1, 1, size=(4, 1))  # 4 đặc trưng
        self.bias = np.random.uniform(-1, 1)
        self.learning_rate = learning_rate
        self.tolerance = tolerance
        self.loss_history = []

    def forward_propagation(self, X):
        """y_predict = Xw + b"""
        return np.dot(X, self.weights) + self.bias

    def cost_function(self, predictions, y):
        """Hàm mất mát (Mean Squared Error - MSE)"""
        return np.mean((y - predictions) ** 2)

    def backward_propagation(self, X, y, predictions):
        """Tính gradient cho trọng số và bias"""
        errors = predictions - y
        dw = 2 * np.dot(X.T, errors) / len(X)
        db = 2 * np.mean(errors)
        return dw, db

    def update_parameters(self, dw, db):
        """Cập nhật trọng số và bias bằng gradient descent"""
        self.weights -= self.learning_rate * dw
        self.bias -= self.learning_rate * db

    def train(self, X, y, iterations=2000):
        """Huấn luyện mô hình với số vòng lặp cụ thể"""
        for i in range(iterations):
            predictions = self.forward_propagation(X)
            cost = self.cost_function(predictions, y)

            # Kiểm tra hội tụ
            if i > 0 and abs(self.loss_history[-1] - cost) < self.tolerance:
                print(f"Gradient Descent hội tụ tại vòng {i + 1}")
                break

            dw, db = self.backward_propagation(X, y, predictions)
            self.update_parameters(dw, db)
            self.loss_history.append(cost)

        print(f"Huấn luyện hoàn tất: trọng số = {self.weights.T}, bias = {self.bias:.4f}")

    def predict(self, X):
        """Dự đoán đầu ra cho đầu vào X"""
        return self.forward_propagation(X)

    def evaluate(self, X, y):
        """Đánh giá mô hình trên tập kiểm tra"""
        predictions = self.predict(X)
        return self.cost_function(predictions, y)

    def plot_loss_curve(self):
        """Vẽ đồ thị mất mát theo vòng lặp"""
        plt.plot(self.loss_history)
        plt.xlabel("Số vòng lặp")
        plt.ylabel("Mất mát")
        plt.title("Đồ thị Mất mát")
        plt.show()

model = LinearRegression(learning_rate=0.0000001, tolerance=1e-6)
model.train(train_input, train_output)

print(model.weights, model.bias)
model.plot_loss_curve()
