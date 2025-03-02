import numpy as np
import matplotlib.pyplot as plt


class ThreeSpiralDataset:
    """
    Tạo tập dữ liệu ba nhánh xoắn ốc (Three Spirals) dùng cho bài toán phân loại.

    Tập dữ liệu bao gồm ba nhóm điểm, mỗi nhóm tạo thành một nhánh của hình xoắn ốc.
    Các điểm được phân bố theo phương trình parametric:

        r = (i / n_points) * R_max
        theta = (i / n_points) * 2 * pi + (j * 2 * pi / n_classes)
        x = r * sin(theta) + noise
        y = r * cos(theta) + noise

    Trong đó:
        - `r` là bán kính của điểm tại vị trí i.
        - `theta` là góc quay của điểm.
        - `R_max` là giá trị bán kính tối đa (ở đây là 5).
        - `noise` là nhiễu Gaussian để tăng tính ngẫu nhiên.

    Attributes:
        n_points (int): Số lượng điểm trên mỗi nhánh.
        n_classes (int): Số lượng nhánh (mặc định là 3).
        noise (float): Độ nhiễu ngẫu nhiên.
        X (numpy.ndarray): Ma trận dữ liệu đầu vào (N x 2).
        y (numpy.ndarray): Nhãn tương ứng (N,).
    """

    def __init__(self, n_points=100, n_classes=3, noise=0.2):
        self.n_points = n_points
        self.n_classes = n_classes
        self.noise = noise
        self.X, self.y = self._generate_spiral()

    def _generate_spiral(self):
        """Tạo dữ liệu hình xoắn ốc."""
        X, y = [], []
        for j in range(self.n_classes):
            ix = np.arange(self.n_points)
            r = ix / self.n_points * 5  # Bán kính
            t = ix / self.n_points * 2 * np.pi + (j * 2 * np.pi / self.n_classes)  # Góc quay
            X.append(np.c_[r * np.sin(t), r * np.cos(t)] + self.noise * np.random.randn(self.n_points, 2))
            y.append(np.full(self.n_points, j))
        return np.vstack(X), np.hstack(y)

    def plot(self):
        """Vẽ tập dữ liệu Three Spirals."""
        plt.figure(figsize=(6, 6))
        plt.scatter(self.X[:, 0], self.X[:, 1], c=self.y, cmap='coolwarm', edgecolors='k')
        plt.title("Three Spirals Dataset")
        plt.show()


class MLPClassifier:
    """
    Mạng nơ-ron nhân tạo (MLP) đơn giản với một lớp ẩn, sử dụng ReLU và Softmax.

    Mô hình sử dụng phương pháp tối ưu Gradient Descent để huấn luyện.

    Attributes:
        d0 (int): Số chiều của dữ liệu đầu vào.
        d1 (int): Số nơ-ron trong lớp ẩn.
        d2 (int): Số lớp đầu ra (số lớp phân loại).
        eta (float): Tốc độ học (learning rate).
        W1, b1: Trọng số và độ chệch của lớp ẩn.
        W2, b2: Trọng số và độ chệch của lớp đầu ra.
    """

    def __init__(self, d0, d1, d2, eta=1):
        self.d0 = d0
        self.d1 = d1
        self.d2 = d2
        self.eta = eta
        self.W1, self.b1, self.W2, self.b2 = self._init_weights()

    def _init_weights(self):
        """Khởi tạo trọng số với giá trị ngẫu nhiên nhỏ."""
        W1 = 0.01 * np.random.randn(self.d0, self.d1)
        b1 = np.zeros(self.d1)
        W2 = 0.01 * np.random.randn(self.d1, self.d2)
        b2 = np.zeros(self.d2)
        return W1, b1, W2, b2

    @staticmethod
    def _softmax_stable(Z):
        """Tính softmax ổn định để tránh tràn số."""
        e_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return e_Z / e_Z.sum(axis=1, keepdims=True)

    @staticmethod
    def _crossentropy_loss(Yhat, y):
        """Tính hàm mất mát cross-entropy.

        Công thức:
            L = -1/N * sum(log(Yhat[i, y_i]))
        """
        id0 = range(Yhat.shape[0])
        return -np.mean(np.log(Yhat[id0, y]))

    def train(self, X, y, epochs=20000):
        """Huấn luyện mô hình bằng thuật toán lan truyền ngược (Backpropagation)."""
        loss_hist = []
        for i in range(epochs):
            # Forward pass
            Z1 = X.dot(self.W1) + self.b1
            A1 = np.maximum(Z1, 0)  # ReLU activation
            Z2 = A1.dot(self.W2) + self.b2
            Yhat = self._softmax_stable(Z2)

            # Compute loss
            if i % 1000 == 0:
                loss = self._crossentropy_loss(Yhat, y)
                print(f"iter {i}, loss: {loss:.6f}")
                loss_hist.append(loss)

            # Backpropagation
            id0 = range(Yhat.shape[0])
            Yhat[id0, y] -= 1
            E2 = Yhat / X.shape[0]

            dW2 = np.dot(A1.T, E2)
            db2 = np.sum(E2, axis=0)
            E1 = np.dot(E2, self.W2.T)
            E1[Z1 <= 0] = 0  # Gradient của ReLU

            dW1 = np.dot(X.T, E1)
            db1 = np.sum(E1, axis=0)

            # Cập nhật trọng số bằng Gradient Descent
            self.W1 -= self.eta * dW1
            self.b1 -= self.eta * db1
            self.W2 -= self.eta * dW2
            self.b2 -= self.eta * db2

        return loss_hist

    def predict(self, X):
        """Dự đoán nhãn của tập dữ liệu X."""
        Z1 = X.dot(self.W1) + self.b1
        A1 = np.maximum(Z1, 0)
        Z2 = A1.dot(self.W2) + self.b2
        return np.argmax(Z2, axis=1)



