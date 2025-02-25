import numpy as np


class SoftmaxRegression:
    """
    Mô hình phân loại Softmax Regression sử dụng thuật toán tối ưu Gradient Descent.

    Mô hình này được sử dụng cho bài toán phân loại nhiều lớp (multi-class classification),
    trong đó hàm mất mát entropy chéo (cross-entropy loss) được tối ưu bằng gradient descent.

    Softmax Regression là một mô hình tuyến tính, áp dụng hàm softmax để tính xác suất
    thuộc về từng lớp và lựa chọn lớp có xác suất cao nhất làm nhãn dự đoán.

    Thuộc tính:
    -----------
    learning_rate : float
        Tốc độ học (learning rate) của thuật toán gradient descent. Giá trị mặc định là 0.01.
    epochs : int
        Số lần lặp (epochs) khi huấn luyện mô hình. Mặc định là 100.
    tol : float
        Ngưỡng dừng sớm (early stopping). Nếu sự thay đổi trọng số nhỏ hơn giá trị này,
        quá trình huấn luyện sẽ dừng sớm. Mặc định là 1e-5.
    batch_size : int
        Số lượng mẫu trong mỗi batch khi sử dụng mini-batch gradient descent. Mặc định là 10.
    W : numpy array
        Ma trận trọng số có kích thước (d, C), trong đó d là số lượng đặc trưng (features)
        và C là số lượng lớp (classes).

    Phương thức:
    ------------
    softmax(Z)
        Tính toán hàm softmax để chuyển đổi đầu ra thành xác suất.
    compute_loss(X, y)
        Tính toán hàm mất mát entropy chéo (cross-entropy loss) trên tập dữ liệu.
    compute_gradient(X, y)
        Tính gradient của hàm mất mát theo trọng số mô hình.
    fit(X, y)
        Huấn luyện mô hình Softmax Regression bằng thuật toán mini-batch gradient descent.
    predict(X)
        Dự đoán nhãn lớp của dữ liệu đầu vào mới.
    """

    def __init__(self, learning_rate=0.01, epochs=100, tol=1e-5, batch_size=10):
        """
        Khởi tạo mô hình Softmax Regression.

        Tham số:
        --------
        learning_rate : float
            Tốc độ học của thuật toán gradient descent.
        epochs : int
            Số lần lặp trong quá trình huấn luyện.
        tol : float
            Ngưỡng sai số dùng để dừng sớm nếu cập nhật trọng số nhỏ hơn giá trị này.
        batch_size : int
            Số lượng mẫu trong mỗi batch khi sử dụng mini-batch gradient descent.
        """
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.tol = tol
        self.batch_size = batch_size
        self.W = None

    def softmax(self, Z):
        """
        Tính toán hàm softmax để chuẩn hóa xác suất của từng lớp.

        Tham số:
        --------
        Z : numpy array
            Ma trận đầu vào có kích thước (N, C), trong đó:
            - N là số lượng mẫu dữ liệu.
            - C là số lượng lớp.

        Trả về:
        -------
        numpy array:
            Ma trận xác suất của từng lớp với kích thước (N, C).
        """
        e_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))  # Giảm thiểu hiện tượng tràn số (stability trick)
        return e_Z / e_Z.sum(axis=1, keepdims=True)

    def compute_loss(self, X, y):
        """
        Tính toán hàm mất mát entropy chéo (cross-entropy loss).

        Tham số:
        --------
        X : numpy array
            Ma trận đặc trưng có kích thước (N, d), trong đó:
            - N là số lượng mẫu dữ liệu.
            - d là số lượng đặc trưng (features).
        y : numpy array
            Mảng nhãn thực tế có kích thước (N,).

        Trả về:
        -------
        float:
            Giá trị mất mát trung bình của toàn bộ dữ liệu.
        """
        A = self.softmax(X.dot(self.W))
        id0 = np.arange(X.shape[0])
        return -np.mean(np.log(A[id0, y]))

    def compute_gradient(self, X, y):
        """
        Tính toán gradient của hàm mất mát với trọng số mô hình.

        Tham số:
        --------
        X : numpy array
            Ma trận đặc trưng có kích thước (N, d).
        y : numpy array
            Mảng nhãn thực tế có kích thước (N,).

        Trả về:
        -------
        numpy array:
            Ma trận gradient có kích thước (d, C).
        """
        A = self.softmax(X.dot(self.W))
        id0 = np.arange(X.shape[0])
        A[id0, y] -= 1
        return X.T.dot(A) / X.shape[0]

    def fit(self, X, y):
        """
        Huấn luyện mô hình Softmax Regression bằng mini-batch gradient descent.

        Tham số:
        --------
        X : numpy array
            Ma trận đặc trưng có kích thước (N, d).
        y : numpy array
            Mảng nhãn thực tế có kích thước (N,).

        Trả về:
        -------
        list:
            Danh sách các giá trị mất mát trong suốt quá trình huấn luyện.
        """
        N, d = X.shape
        C = np.max(y) + 1  # Số lượng lớp
        self.W = np.random.randn(d, C)
        W_old = self.W.copy()
        loss_hist = [self.compute_loss(X, y)]

        nbatches = int(np.ceil(N / self.batch_size))
        for ep in range(self.epochs):
            mix_ids = np.random.permutation(N)
            for i in range(nbatches):
                batch_ids = mix_ids[self.batch_size * i: min(self.batch_size * (i + 1), N)]
                X_batch, y_batch = X[batch_ids], y[batch_ids]
                self.W -= self.learning_rate * self.compute_gradient(X_batch, y_batch)
            loss_hist.append(self.compute_loss(X, y))

            if np.linalg.norm(self.W - W_old) / self.W.size < self.tol:
                break
            W_old = self.W.copy()

        return loss_hist

    def predict(self, X):
        """
        Dự đoán nhãn của dữ liệu đầu vào.

        Tham số:
        --------
        X : numpy array
            Ma trận đặc trưng có kích thước (N, d).

        Trả về:
        -------
        numpy array:
            Mảng nhãn dự đoán có kích thước (N,).
        """
        return np.argmax(X.dot(self.W), axis=1)
