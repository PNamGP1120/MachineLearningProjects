import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class KNNClassifier:
    """
    Bộ phân loại K-Nearest Neighbors (KNN) cho dữ liệu hoa IRIS.

    📌 **Ý tưởng thuật toán**:
    - Tìm `k` điểm gần nhất với điểm dữ liệu cần dự đoán.
    - Sử dụng `mode()` để chọn nhãn xuất hiện nhiều nhất trong `k` điểm lân cận.
    - Khoảng cách Euclidean được sử dụng để tính độ gần giữa các điểm.

    📌 **Công thức toán học**:
    Khoảng cách Euclidean giữa hai điểm `X` và `Y` trong không gian `n` chiều:


    d(X, Y) = \sqrt{\sum_{i=1}^{n} (X_i - Y_i)^2}

    """

    def __init__(self, k=5):
        """
        Khởi tạo bộ phân loại KNN.

        📌 **Tham số**:
        - `k` (int): Số lượng láng giềng gần nhất để xem xét khi phân loại.
        """
        self.k = k
        self.train_set = None
        self.test_set = None

    def load_data(self, file_path):
        """
        Tải dữ liệu từ tệp CSV và chia thành tập huấn luyện và tập kiểm tra.

        📌 **Tham số**:
        - `file_path` (str): Đường dẫn đến tệp CSV chứa dữ liệu IRIS.

        📌 **Ghi chú**:
        - Dữ liệu được xáo trộn (`shuffle`) để đảm bảo tính ngẫu nhiên.
        - 2/3 dữ liệu được dùng để huấn luyện, 1/3 còn lại dùng để kiểm tra.
        """
        data = pd.read_csv(file_path).sample(frac=1, random_state=42)  # Trộn ngẫu nhiên dữ liệu
        self.train_set = data.iloc[:len(data) * 2 // 3]
        self.test_set = data.iloc[len(data) * 2 // 3 + 1:]

    def _euclidean_distance(self, X, Y):
        """
        Tính khoảng cách Euclidean giữa hai điểm `X` và `Y`.

        📌 **Tham số**:
        - `X`, `Y` (numpy array hoặc pandas Series): Hai điểm cần tính khoảng cách.

        📌 **Trả về**:
        - `float`: Khoảng cách Euclidean giữa `X` và `Y`.
        """
        return np.sqrt(np.sum((X - Y) ** 2))

    def predict(self, data_point):
        """
        Dự đoán nhãn của một điểm dữ liệu mới bằng thuật toán KNN.

        📌 **Tham số**:
        - `data_point` (numpy array hoặc pandas Series): Dữ liệu của điểm cần dự đoán.

        📌 **Cách hoạt động**:
        - Tính khoảng cách giữa `data_point` và tất cả các điểm trong tập huấn luyện.
        - Lấy `k` điểm gần nhất.
        - Chọn nhãn xuất hiện nhiều nhất trong `k` điểm đó.

        📌 **Trả về**:
        - `str`: Nhãn dự đoán của điểm dữ liệu.
        """
        train_features = self.train_set.drop(columns=['species'])
        distances = train_features.apply(lambda row: self._euclidean_distance(row, data_point), axis=1)
        k_nearest_labels = self.train_set.loc[distances.nsmallest(self.k).index, 'species']
        return k_nearest_labels.mode()[0]  # Chọn nhãn xuất hiện nhiều nhất

    def evaluate(self):
        """
        Đánh giá độ chính xác của mô hình trên tập kiểm tra.

        📌 **Cách hoạt động**:
        - Chạy `predict()` trên tất cả các mẫu trong tập kiểm tra.
        - Đếm số lần dự đoán đúng.
        - Tính toán độ chính xác.

        📌 **Trả về**:
        - `float`: Độ chính xác của mô hình tính theo phần trăm.
        """
        test_features = self.test_set.drop(columns=['species'])
        correct_predictions = sum(self.predict(row) == self.test_set.loc[index, 'species']
                                  for index, row in test_features.iterrows())

        return correct_predictions * 100 / len(self.test_set)

    def plot_decision_boundary(self):
        """
        Vẽ biểu đồ thể hiện kết quả phân loại của thuật toán KNN.

        📌 **Ý tưởng**:
        - Hiển thị tập dữ liệu huấn luyện (dưới dạng điểm màu).
        - Hiển thị ranh giới quyết định bằng cách kiểm tra từng điểm trên lưới.

        📌 **Ghi chú**:
        - Chỉ áp dụng cho dữ liệu có 2 đặc trưng (2D).
        """
        if self.train_set.shape[1] - 1 != 2:
            print("⚠️ Không thể vẽ biểu đồ cho dữ liệu có nhiều hơn 2 đặc trưng.")
            return

        train_features = self.train_set.drop(columns=['species']).values
        train_labels = self.train_set['species'].values

        # Xác định phạm vi của trục x và y
        x_min, x_max = train_features[:, 0].min() - 1, train_features[:, 0].max() + 1
        y_min, y_max = train_features[:, 1].min() - 1, train_features[:, 1].max() + 1

        # Tạo lưới điểm
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                             np.linspace(y_min, y_max, 100))

        # Dự đoán lớp cho từng điểm trong lưới
        Z = np.array([self.predict(np.array([x, y])) for x, y in np.c_[xx.ravel(), yy.ravel()]])
        Z = Z.reshape(xx.shape)

        # Vẽ ranh giới quyết định
        plt.contourf(xx, yy, Z, alpha=0.3)

        # Vẽ dữ liệu huấn luyện
        species_map = {'Iris-setosa': 'red', 'Iris-versicolor': 'blue', 'Iris-virginica': 'green'}
        for species, color in species_map.items():
            subset = self.train_set[self.train_set['species'] == species]
            plt.scatter(subset.iloc[:, 0], subset.iloc[:, 1], c=color, label=species, edgecolors='k')

        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title(f'Biểu đồ phân loại KNN (k={self.k})')
        plt.legend()
        plt.show()


# 🚀 **Chạy thử nghiệm KNN**
knn = KNNClassifier(k=5)
knn.load_data('iris_2features.csv')

# Đánh giá độ chính xác
accuracy = knn.evaluate()
print(f'🎯 Độ chính xác của mô hình KNN với k=5: {accuracy:.2f}%')

# Vẽ biểu đồ kết quả
knn.plot_decision_boundary()
