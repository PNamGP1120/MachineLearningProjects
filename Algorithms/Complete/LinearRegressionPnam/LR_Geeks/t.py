import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self):
        self.w = None

    def fit(self, X, y):
        XTX = np.linalg.pinv(X.T @ X)
        XTy = X.T @ y
        self.w = XTX @ XTy

    def predict(self, X):
        return X @ self.w


def plot_results(X, y, y_pred):
    """
    Vẽ biểu đồ so sánh giữa dữ liệu thực tế và dự đoán của mô hình.

    Nếu dữ liệu có 1 đặc trưng (n = 1):
        - Vẽ scatter plot giữa X và y.
        - Vẽ đường hồi quy tuyến tính (dự đoán).

    Nếu dữ liệu có 2 đặc trưng (n = 2) trở lên:
        - Chỉ hiển thị kết quả theo đặc trưng đầu tiên (X[:, 0]).

    Parameters:
        X (numpy.ndarray): Ma trận đầu vào có kích thước (m x n).
        y (numpy.ndarray): Vector giá trị thực tế (m,).
        y_pred (numpy.ndarray): Vector giá trị dự đoán (m,).

    Returns:
        None (Hiển thị biểu đồ matplotlib).
    """
    plt.figure(figsize=(8, 6))

    if X.shape[1] == 1:  # Nếu có 1 đặc trưng, vẽ theo X[:, 0]
        plt.scatter(X, y, color='blue', label='Thực tế')
        plt.plot(X, y_pred, color='red', linewidth=2, label='Dự đoán')

    else:  # Nếu có nhiều hơn 1 đặc trưng, chỉ vẽ theo đặc trưng đầu tiên
        plt.scatter(X[:, 0], y, color='blue', label='Thực tế')
        plt.scatter(X[:, 0], y_pred, color='red', label='Dự đoán', alpha=0.7)

    # Vẽ đường sai số
    for i in range(len(y)):
        plt.plot([X[i, 0], X[i, 0]], [y[i], y_pred[i]], color='gray', linestyle='dotted')

    plt.xlabel("Đặc trưng X (cột đầu tiên)")
    plt.ylabel("Giá trị y")
    plt.title("So sánh Thực tế vs Dự đoán")
    plt.legend()
    plt.show()


# Test hàm vẽ
def test_model():
    """
    Chạy kiểm thử mô hình, huấn luyện trên dữ liệu giả lập và vẽ kết quả.
    """
    np.random.seed(42)
    X = np.random.rand(100, 1) * 10  # 100 mẫu, 1 đặc trưng (giá trị từ 0 đến 10)
    w_true = np.array([3])  # Trọng số thực tế
    noise = np.random.randn(100) * 0.5  # Thêm nhiễu Gaussian
    y = X @ w_true + noise  # Công thức tuyến tính có nhiễu

    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)

    print("Trọng số thực tế:", w_true)
    print("Trọng số mô hình học được:", model.w)

    # Vẽ kết quả
    plot_results(X, y, y_pred)


# Chạy kiểm thử và vẽ kết quả
test_model()
