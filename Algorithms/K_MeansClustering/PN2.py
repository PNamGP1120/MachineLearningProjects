import numpy as np
import matplotlib.pyplot as plt

class KMeansClustering:
    """
    Thuật toán K-Means Clustering để phân nhóm dữ liệu.

    - Chọn ngẫu nhiên K điểm làm tâm cụm (centroids).
    - Gán mỗi điểm dữ liệu vào cụm có tâm gần nhất.
    - Cập nhật lại vị trí tâm cụm dựa trên trung bình của các điểm trong cụm.
    - Lặp lại cho đến khi không có sự thay đổi giữa các vòng lặp.

    Attributes:
        num_clusters (int): Số cụm cần phân nhóm.
        num_points (int): Số điểm dữ liệu.
        datapoints (np.ndarray): Tập hợp dữ liệu đầu vào.
        clusters (list): Danh sách các cụm chứa centroid, datapoints và nhãn.
    """

    def __init__(self, num_clusters: int, num_points: int):
        """
        Khởi tạo thuật toán K-Means với số cụm và số điểm dữ liệu.

        :param num_clusters: Số lượng cụm K.
        :param num_points: Số lượng điểm dữ liệu N.
        """
        self.num_clusters = num_clusters
        self.num_points = num_points
        self.datapoints = self._create_datapoints(num_points)
        self.clusters = self._initialize_clusters()

    def _create_datapoints(self, num_points: int):
        """
        Tạo ngẫu nhiên N điểm dữ liệu trong không gian 2D.

        :param num_points: Số điểm dữ liệu cần tạo.
        :return: Mảng numpy chứa tọa độ các điểm dữ liệu.
        """
        possible_points = [(x, y) for x in range(num_points + 1) for y in range(num_points + 1)]
        unique_points = np.random.choice(len(possible_points), num_points, replace=False)
        return np.array([possible_points[i] for i in unique_points])

    def _initialize_clusters(self):
        """
        Khởi tạo K cụm bằng cách chọn ngẫu nhiên các điểm từ dữ liệu.

        :return: Danh sách các cụm với centroid ban đầu.
        """
        centroids = self.datapoints[np.random.choice(self.datapoints.shape[0], size=self.num_clusters, replace=False)]
        return [{'centroid': centroid, 'datapoints': np.empty((0, 2), dtype=int), 'label': ""} for centroid in centroids]

    @staticmethod
    def _distance(p1, p2):
        """
        Tính khoảng cách Euclidean giữa hai điểm.

        :param p1: Điểm đầu tiên.
        :param p2: Điểm thứ hai.
        :return: Khoảng cách giữa hai điểm.
        """
        return np.linalg.norm(p1 - p2)

    def _assign_clusters(self):
        """
        Gán từng điểm dữ liệu vào cụm gần nhất.

        :return: Danh sách các cụm đã cập nhật với điểm dữ liệu mới.
        """
        for cluster in self.clusters:
            cluster['datapoints'] = np.empty((0, 2), dtype=int)  # Reset điểm dữ liệu

        for datapoint in self.datapoints:
            nearest_index = np.argmin([self._distance(cluster['centroid'], datapoint) for cluster in self.clusters])
            self.clusters[nearest_index]['datapoints'] = np.append(self.clusters[nearest_index]['datapoints'], [datapoint], axis=0)

    def _update_clusters(self):
        """
        Cập nhật lại vị trí tâm cụm dựa trên trung bình của các điểm dữ liệu.

        :return: True nếu có sự thay đổi trong vị trí tâm cụm, ngược lại False.
        """
        old_centroids = [cluster['centroid'].copy() for cluster in self.clusters]

        for cluster in self.clusters:
            if cluster['datapoints'].size > 0:
                cluster['centroid'] = np.mean(cluster['datapoints'], axis=0)

        return not np.allclose(old_centroids, [cluster['centroid'] for cluster in self.clusters], atol=1e-3)

    def train(self):
        """
        Thực thi thuật toán K-Means cho đến khi hội tụ.

        :return: Số vòng lặp để thuật toán hội tụ.
        """
        iteration = 0
        while True:
            iteration += 1
            self._assign_clusters()
            if not self._update_clusters():
                break
        return iteration

    def print_clusters(self):
        """
        Hiển thị thông tin các cụm sau khi huấn luyện.
        """
        for i, cluster in enumerate(self.clusters):
            print(f"Cluster {i + 1}:")
            print(f"  Centroid: {cluster['centroid']}")
            print(f"  Datapoints: {cluster['datapoints'].shape[0]} points")
            print(f"  Label: {cluster['label']}\n")

    def draw_clusters(self):
        """
        Vẽ biểu đồ trực quan của các cụm sau khi huấn luyện.
        """
        plt.figure(figsize=(8, 6))
        colors = plt.get_cmap('tab10', self.num_clusters)

        for idx, cluster in enumerate(self.clusters):
            if cluster['datapoints'].size > 0:
                plt.scatter(cluster['datapoints'][:, 0], cluster['datapoints'][:, 1], color=colors(idx), label=f'Cluster {idx + 1}')
                plt.scatter(cluster['centroid'][0], cluster['centroid'][1], marker='*', color=colors(idx), s=200, edgecolor='black')

        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("K-Means Clustering")
        plt.legend()
        plt.grid(True)
        plt.show()


if __name__ == '__main__':
    # Số lượng điểm dữ liệu và số cụm
    num_points = 100
    num_clusters = 3

    # Khởi tạo và chạy K-Means
    kmeans = KMeansClustering(num_clusters, num_points)
    iterations = kmeans.train()

    # Hiển thị kết quả
    print(f"K-Means hội tụ sau {iterations} vòng lặp.\n")
    kmeans.print_clusters()
    kmeans.draw_clusters()
