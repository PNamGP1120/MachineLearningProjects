import numpy as np
import matplotlib.pyplot as plt

def create_N_datapoints(num_points: int, dim: int):
    """
    Tạo N điểm dữ liệu với số chiều dim (1D, 2D, 3D).
    """
    return np.random.rand(num_points, dim) * 10  # Tạo giá trị ngẫu nhiên từ 0 đến 10

def create_K_clusters(datapoints, K_centroids):
    """
    Chọn K tâm cụm ngẫu nhiên từ dữ liệu.
    """
    centroids = datapoints[np.random.choice(datapoints.shape[0], K_centroids, replace=False)]
    return [{'centroid': centroid, 'datapoints': np.empty((0, datapoints.shape[1])), 'label': ""} for centroid in centroids]

def distance(p1, p2):
    """
    Tính khoảng cách Euclidean giữa hai điểm p1 và p2.
    """
    return np.linalg.norm(p1 - p2)

def datapoints_assign_clusters(datapoints, clusters):
    """
    Gán mỗi điểm dữ liệu vào cụm gần nhất.
    """
    for cluster in clusters:
        cluster['datapoints'] = np.empty((0, datapoints.shape[1]))  # Reset lại danh sách điểm

    for datapoint in datapoints:
        nearest_idx = np.argmin([distance(cluster['centroid'], datapoint) for cluster in clusters])
        clusters[nearest_idx]['datapoints'] = np.vstack([clusters[nearest_idx]['datapoints'], datapoint])

    return clusters

def updateClusters(datapoints, clusters):
    """
    Cập nhật tâm cụm bằng trung bình tất cả các điểm trong cụm.
    """
    converged = True  # Kiểm tra nếu tâm cụm không thay đổi
    for cluster in clusters:
        if len(cluster['datapoints']) > 0:
            new_centroid = np.mean(cluster['datapoints'], axis=0)
            if not np.array_equal(new_centroid, cluster['centroid']):
                converged = False
            cluster['centroid'] = new_centroid

    datapoints_assign_clusters(datapoints, clusters)  # Gán lại cụm
    return clusters, converged

def drawClusters(clusters, dim, iteration):
    """
    Vẽ đồ thị cụm dữ liệu tương ứng với số chiều dim (1D, 2D, 3D).
    """
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d" if dim == 3 else None)
    plt.title(f'K-Means Clustering - Iteration {iteration}')

    colors = plt.colormaps['tab10']

    for idx, cluster in enumerate(clusters):
        if cluster['datapoints'].size > 0:
            if dim == 1:
                ax.scatter(cluster['datapoints'], np.zeros_like(cluster['datapoints']), color=colors(idx), label=f'Cluster {idx + 1}')
                ax.scatter(cluster['centroid'], 0, marker='*', color=colors(idx), s=200, edgecolor='white')
            elif dim == 2:
                ax.scatter(cluster['datapoints'][:, 0], cluster['datapoints'][:, 1], color=colors(idx), label=f'Cluster {idx + 1}')
                ax.scatter(cluster['centroid'][0], cluster['centroid'][1], marker='*', color=colors(idx), s=200, edgecolor='white')
            elif dim == 3:
                ax.scatter(cluster['datapoints'][:, 0], cluster['datapoints'][:, 1], cluster['datapoints'][:, 2], color=colors(idx), label=f'Cluster {idx + 1}')
                ax.scatter(cluster['centroid'][0], cluster['centroid'][1], cluster['centroid'][2], marker='*', color=colors(idx), s=200, edgecolor='white')

    plt.legend()
    plt.show()

if __name__ == '__main__':
    num_points = 10  # Số lượng điểm dữ liệu
    dim = 3  # Chọn số chiều: 1, 2 hoặc 3
    K_centroids = 3  # Số cụm

    datapoints = create_N_datapoints(num_points, dim)
    clusters = create_K_clusters(datapoints, K_centroids)
    clusters = datapoints_assign_clusters(datapoints, clusters)

    iteration = 0
    while True:
        iteration += 1
        drawClusters(clusters, dim, iteration)
        clusters, converged = updateClusters(datapoints, clusters)
        if converged:
            break
