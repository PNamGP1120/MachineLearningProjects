import numpy as np
import matplotlib.pyplot as plt

class LogisticRegression:
    """
    Mô hình Logistic Regression sử dụng Stochastic Gradient Descent (SGD) để tối ưu hóa trọng số.

    Thuật toán huấn luyện sử dụng Binary Cross-Entropy Loss kèm Regularization L2 để giảm overfitting.

    Attributes:
        X (numpy.ndarray): Ma trận đầu vào (N, d) với cột đầu tiên là bias (toàn bộ là 1).
        y (numpy.ndarray): Vector nhãn (N,), chứa giá trị 0 hoặc 1.
        weights (numpy.ndarray): Vector trọng số (d+1,), khởi tạo ngẫu nhiên.
        loss_hist (list): Danh sách chứa giá trị loss của từng epoch trong quá trình huấn luyện.

    Methods:
        sigmoid(z): Tính sigmoid của giá trị đầu vào.
        prob(): Dự đoán xác suất P(y=1|X).
        loss(lamda): Tính hàm loss Binary Cross-Entropy có Regularization.
        train(lamda=0.001, lr=0.01, nepochs=100): Huấn luyện mô hình bằng SGD.
        predict(X_test): Dự đoán nhãn (0 hoặc 1) của dữ liệu đầu vào.
        plot_decision_boundary(): Vẽ decision boundary (chỉ hỗ trợ dữ liệu 1D).
    """

    def __init__(self, X, y):
        """
        Khởi tạo mô hình Logistic Regression.

        Parameters:
            X (numpy.ndarray): Ma trận đặc trưng đầu vào có kích thước (N, d).
            y (numpy.ndarray): Vector nhãn có kích thước (N,), chứa giá trị 0 hoặc 1.

        Biến đổi dữ liệu:
            - Thêm một cột bias (1) vào ma trận X.
            - Khởi tạo trọng số ngẫu nhiên nhỏ để tránh gradient vanishing.
        """
        self.X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)  # Thêm bias
        self.y = y
        self.weights = np.random.randn(self.X.shape[1]) * 0.01  # Trọng số nhỏ ngẫu nhiên
        self.loss_hist = []  # Lưu lịch sử loss

    def sigmoid(self, z):
        """
        Hàm kích hoạt sigmoid.

        Parameters:
            z (numpy.ndarray): Giá trị đầu vào.

        Returns:
            numpy.ndarray: Giá trị sigmoid của z.
        """
        return 1 / (1 + np.exp(-z))

    def prob(self):
        """
        Dự đoán xác suất P(y=1|X) theo mô hình Logistic Regression.

        Returns:
            numpy.ndarray: Xác suất dự đoán của từng mẫu.
        """
        return self.sigmoid(self.X @ self.weights)

    def loss(self, lamda):
        """
        Tính hàm mất mát Binary Cross-Entropy có Regularization.

        Parameters:
            lamda (float): Hệ số regularization (L2).

        Returns:
            float: Giá trị của hàm loss.
        """
        p = self.prob()
        eps = 1e-8  # Tránh log(0)
        return np.mean(-self.y * np.log(p + eps) - (1 - self.y) * np.log(1 - p + eps)) + lamda * np.sum(
            self.weights ** 2) / 2

    def train(self, lamda=0.001, lr=0.01, nepochs=100):
        """
        Huấn luyện mô hình Logistic Regression bằng Stochastic Gradient Descent (SGD).

        Parameters:
            lamda (float): Hệ số regularization (L2). Mặc định = 0.001.
            lr (float): Tốc độ học (learning rate). Mặc định = 0.01.
            nepochs (int): Số epoch huấn luyện. Mặc định = 100.

        Returns:
            numpy.ndarray: Trọng số sau khi huấn luyện.

        Cập nhật trọng số theo công thức:
            w = w - lr * [(p - y) * x + lamda * w]
        """
        N, d = self.X.shape
        self.loss_hist = [self.loss(lamda)]  # Lưu lịch sử loss

        for ep in range(nepochs):
            mix_ids = np.random.permutation(N)  # Xáo trộn dữ liệu
            for i in mix_ids:
                xi = self.X[i]  # Lấy một mẫu
                yi = self.y[i]
                zi = self.sigmoid(np.dot(xi, self.weights))  # Dự đoán xác suất
                self.weights -= lr * ((zi - yi) * xi + lamda * self.weights)  # SGD update

            self.loss_hist.append(self.loss(lamda))  # Lưu loss mỗi epoch

            # Dừng sớm nếu loss thay đổi rất nhỏ
            if len(self.loss_hist) > 1 and abs(self.loss_hist[-1] - self.loss_hist[-2]) < 1e-15:
                break

        return self.weights

    def predict(self, X_test):
        """
        Dự đoán nhãn (0 hoặc 1) cho dữ liệu mới.

        Parameters:
            X_test (numpy.ndarray): Ma trận đặc trưng đầu vào có kích thước (M, d).

        Returns:
            numpy.ndarray: Vector nhãn dự đoán (M,), chứa giá trị 0 hoặc 1.
        """
        X_test = np.concatenate((np.ones((X_test.shape[0], 1)), X_test), axis=1)  # Thêm bias
        return (self.sigmoid(X_test @ self.weights) >= 0.5).astype(int)  # Ngưỡng 0.5

    def plot_decision_boundary(self):
        """
        Vẽ Decision Boundary (chỉ hỗ trợ cho dữ liệu có 1 đặc trưng).

        Nếu dữ liệu có nhiều hơn 1 đặc trưng, sẽ thông báo lỗi.
        """
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

