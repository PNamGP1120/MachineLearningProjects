import numpy as np
import matplotlib.pyplot as plt

class PerceptronRegression:
    """
    Mô hình hồi quy Perceptron sử dụng thuật toán cập nhật trọng số để phân loại dữ liệu tuyến tính.

    Phương pháp này tìm kiếm một siêu phẳng (hyperplane) có thể phân chia hai lớp dữ liệu
    bằng cách cập nhật trọng số dựa trên điểm bị phân loại sai.

    Attributes:
    -----------
    w : numpy array
        Trọng số của mô hình, được cập nhật trong quá trình huấn luyện.
    """

    def __init__(self):
        """
        Khởi tạo mô hình với trọng số ban đầu là None.
        """
        self.w = None

    def predict(self, X):
        """
        Dự đoán nhãn của tập dữ liệu đầu vào.

        Tham số:
        --------
        X : numpy array, shape (N, d)
            Ma trận đặc trưng với N mẫu, mỗi mẫu có d đặc trưng.

        Trả về:
        -------
        numpy array
            Mảng nhãn dự đoán (1 hoặc -1).
        """
        return np.sign(X.dot(self.w))

    def fit(self, X, y, w_init=None):
        """
        Huấn luyện mô hình Perceptron bằng cách tìm trọng số phù hợp.

        Tham số:
        --------
        X : numpy array, shape (N, d)
            Ma trận đặc trưng của N mẫu, mỗi mẫu có d đặc trưng.
        y : numpy array, shape (N,)
            Mảng nhãn đầu ra (-1 hoặc 1).
        w_init : numpy array, optional
            Trọng số khởi tạo ban đầu. Nếu không cung cấp, sẽ được khởi tạo ngẫu nhiên.

        Trả về:
        -------
        numpy array
            Trọng số sau khi huấn luyện.
        """
        N, d = X.shape
        self.w = w_init if w_init is not None else np.random.randn(d)

        while True:
            pred = self.predict(X)

            # Tìm các điểm bị phân loại sai
            misclassified_idxs = np.where(pred != y)[0]

            # Nếu không còn điểm nào bị phân loại sai, dừng thuật toán
            if len(misclassified_idxs) == 0:
                break

            # Chọn ngẫu nhiên một điểm bị phân loại sai để cập nhật trọng số
            random_idx = np.random.choice(misclassified_idxs)
            self.w += y[random_idx] * X[random_idx]

        return self.w

    def decision_boundary(self, X):
        """
        Tính toán đường quyết định của mô hình.

        Tham số:
        --------
        X : numpy array, shape (N, d)
            Ma trận đặc trưng.

        Trả về:
        -------
        numpy array
            Mảng giá trị x2 tương ứng với x1 để vẽ đường quyết định.
        """
        x1_range = np.linspace(X[:, 1].min(), X[:, 1].max(), 100)
        x2_range = -(self.w[0] + self.w[1] * x1_range) / self.w[2]
        return x1_range, x2_range


# ========================== KIỂM THỬ MÔ HÌNH ========================== #

# Tạo dữ liệu hai lớp
means = [[-1, 0], [1, 0]]  # Tọa độ trung tâm của hai cụm dữ liệu
cov = [[0.3, 0.2], [0.2, 0.3]]  # Ma trận hiệp phương sai
N = 50  # Số lượng điểm dữ liệu mỗi lớp

X0 = np.random.multivariate_normal(means[0], cov, N)  # Lớp 1
X1 = np.random.multivariate_normal(means[1], cov, N)  # Lớp -1

X = np.concatenate((X0, X1), axis=0)  # Ghép hai lớp vào một tập dữ liệu
y = np.concatenate((np.ones(N), -1 * np.ones(N)))  # Nhãn: Lớp 1 -> 1, Lớp -1 -> -1

# Bổ sung cột bias (1) vào X để sử dụng với trọng số w
Xbar = np.concatenate((np.ones((2 * N, 1)), X), axis=1)

# Khởi tạo mô hình và huấn luyện
model = PerceptronRegression()
w_init = np.random.randn(Xbar.shape[1])  # Trọng số khởi tạo ngẫu nhiên
model.fit(Xbar, y, w_init)

# Vẽ dữ liệu
plt.figure(figsize=(6, 6))
plt.scatter(X0[:, 0], X0[:, 1], color="blue", label="Class +1", edgecolors="k")
plt.scatter(X1[:, 0], X1[:, 1], color="red", label="Class -1", edgecolors="k")

# Vẽ đường quyết định
x1_range, x2_range = model.decision_boundary(Xbar)
plt.plot(x1_range, x2_range, "g--", label="Decision Boundary")

# Cấu hình đồ thị
plt.xlim(-2, 2)
plt.ylim(-2, 2)
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.title("Perceptron Decision Boundary")
plt.legend()
plt.grid()
plt.show()