import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

np.random.seed(18)

# Tạo dữ liệu giả lập
means = [[2, 2], [8, 3], [3, 6]]
cov = [[1, 0], [0, 1]]
N = 500
X0 = np.random.multivariate_normal(means[0], cov, N)
X1 = np.random.multivariate_normal(means[1], cov, N)
X2 = np.random.multivariate_normal(means[2], cov, N)
X = np.concatenate((X0, X1, X2), axis=0)
K = 3  # 3 cụm
original_label = np.asarray([0]*N + [1]*N + [2]*N).T

# Áp dụng KMeans từ scikit-learn
model = KMeans(n_clusters=K, random_state=0).fit(X)
print('Centers found by scikit-learn:')
print(model.cluster_centers_)
pred_label = model.predict(X)


# Hàm vẽ K-Means
def kmeans_display(X, labels, centroids):
    plt.figure(figsize=(10, 8))
    K = len(np.unique(labels))
    colors = plt.get_cmap('tab10', K)

    for k in range(K):
        cluster_data = X[labels == k]
        plt.scatter(cluster_data[:, 0], cluster_data[:, 1], label=f'Cluster {k + 1}', color=colors(k))

    # Vẽ tâm cụm
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='*', color='red', s=200, edgecolor='white', label='Centroids')
    plt.legend()
    plt.title('K-Means Clustering Result')
    plt.show()


# Chuyển pred_label thành numpy array để tiện lọc
kmeans_display(X, np.array(pred_label), model.cluster_centers_)
