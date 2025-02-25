import numpy as np
from model import SoftmaxRegression

def generate_data(n_samples=500, n_features=10, n_classes=3):
    """
    Tạo dữ liệu giả lập gồm nhiều lớp bằng cách lấy mẫu từ phân phối chuẩn.

    Tham số:
    - n_samples (int): Số lượng mẫu dữ liệu.
    - n_features (int): Số lượng đặc trưng.
    - n_classes (int): Số lượng lớp.

    Trả về:
    - X (numpy array): Ma trận đặc trưng (n_samples, n_features).
    - y (numpy array): Nhãn lớp (n_samples,).
    """
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features)  # Dữ liệu ngẫu nhiên từ phân phối chuẩn
    y = np.random.randint(0, n_classes, size=n_samples)  # Nhãn ngẫu nhiên từ 0 đến n_classes-1
    return X, y

def train_test_split(X, y, test_size=0.2):
    """
    Chia dữ liệu thành tập huấn luyện và tập kiểm tra.

    Tham số:
    - X (numpy array): Ma trận đặc trưng.
    - y (numpy array): Nhãn lớp.
    - test_size (float): Tỷ lệ tập kiểm tra.

    Trả về:
    - X_train, X_test, y_train, y_test
    """
    n_samples = X.shape[0]
    test_samples = int(n_samples * test_size)

    indices = np.random.permutation(n_samples)
    test_indices = indices[:test_samples]
    train_indices = indices[test_samples:]

    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]

def normalize_data(X):
    """
    Chuẩn hóa dữ liệu bằng cách trừ trung bình và chia độ lệch chuẩn.

    Tham số:
    - X (numpy array): Ma trận đặc trưng.

    Trả về:
    - X chuẩn hóa
    """
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return (X - mean) / (std + 1e-8)  # Tránh chia cho 0

def compute_accuracy(y_true, y_pred):
    """
    Tính độ chính xác của mô hình.

    Tham số:
    - y_true (numpy array): Nhãn thực tế.
    - y_pred (numpy array): Nhãn dự đoán.

    Trả về:
    - Độ chính xác (%)
    """
    return np.mean(y_true == y_pred) * 100

def test_softmax_regression():
    """
    Kiểm thử thuật toán Softmax Regression trên tập dữ liệu giả lập.
    """
    # 1. Tạo dữ liệu
    X, y = generate_data(n_samples=500, n_features=10, n_classes=3)

    # 2. Chia dữ liệu thành tập huấn luyện và tập kiểm tra
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # 3. Chuẩn hóa dữ liệu
    X_train = normalize_data(X_train)
    X_test = normalize_data(X_test)

    # 4. Huấn luyện mô hình
    model = SoftmaxRegression(learning_rate=0.1, epochs=300, batch_size=20)
    model.fit(X_train, y_train)

    # 5. Dự đoán trên tập kiểm tra
    y_pred = model.predict(X_test)

    # 6. Đánh giá độ chính xác
    accuracy = compute_accuracy(y_test, y_pred)

    print(f"Độ chính xác trên tập kiểm tra: {accuracy:.2f}%")

# Chạy kiểm thử
test_softmax_regression()
