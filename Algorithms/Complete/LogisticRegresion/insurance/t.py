import numpy as np
from model import LogisticRegression


def test_logistic_regression():
    """
    Kiểm tra mô hình Logistic Regression bằng cách tạo dữ liệu giả lập, huấn luyện mô hình và vẽ kết quả.

    Quá trình kiểm tra bao gồm:
        1. Tạo dữ liệu ngẫu nhiên (dữ liệu tuyến tính có nhiễu).
        2. Chia tập dữ liệu thành tập huấn luyện và kiểm tra.
        3. Huấn luyện mô hình Logistic Regression.
        4. Dự đoán nhãn cho tập kiểm tra.
        5. Tính độ chính xác (accuracy).
        6. Vẽ Decision Boundary nếu dữ liệu có 1 feature.
    """
    # Tạo dữ liệu ngẫu nhiên
    np.random.seed(42)  # Đảm bảo kết quả có thể lặp lại
    N = 100  # Số lượng mẫu
    X = np.random.randn(N, 1) * 2  # Dữ liệu có 1 feature, nhân 2 để tăng độ phân tán
    y = (X + np.random.randn(N, 1) > 0).astype(int).flatten()  # Nhãn y = 1 nếu x + nhiễu > 0

    # Chia tập train/test (80% train, 20% test)
    split = int(0.8 * N)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Khởi tạo và huấn luyện mô hình
    model = LogisticRegression(X_train, y_train)
    model.train(lamda=0.001, lr=0.1, nepochs=500)

    # Dự đoán trên tập kiểm tra
    y_pred = model.predict(X_test)

    # Tính độ chính xác
    accuracy = np.mean(y_pred == y_test)
    print(f"Accuracy on test set: {accuracy * 100:.2f}%")

    # Vẽ Decision Boundary nếu có thể
    model.plot_decision_boundary()

# Gọi hàm test
test_logistic_regression()
