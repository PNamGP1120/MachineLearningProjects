import numpy as np
import matplotlib.pyplot as plt

def create_N_datapoints(num_points: int):
    """
    Create N datapoints from a data frame

    :type num_points: int
    :return N: number of datapoints
    """
    possible_points = [(x, y) for x in range(num_points + 1) for y in range(num_points + 1)]
    unique_points = np.random.choice(len(possible_points), num_points, replace=False)
    return np.array([possible_points[i] for i in unique_points])

#tao K centroid
def create_K_clusters(int_datapoint, K_centroids):
    """
    return
    [
    {
        centroid: [1, 2];
        datapoints: [[]];
        labels: ""
    }
    ]
    :type int_datapoint: np.ndarray
    :type K_centroids: int
    """

    # chon ra K diem lam centroid tu datapoints
    centroids = int_datapoint[np.random.choice(int_datapoint.shape[0], size=K_centroids, replace=False)]
    return [{'centroid': centroid, 'datapoints':np.empty((0,2), dtype=int), 'label':"" } for centroid in centroids]


def distance(p1, p2):
    """

    :type p1: np.ndarray
    :type p2: np.ndarray
    """
    return np.linalg.norm(p1 - p2)

def datapoints_assign_clusters(datapoints, clusters):
    """

    :type datapoints: np.ndarray
    :type clusters: object
    """
    for centroid in clusters:
        centroid['datapoints'] = np.empty((0,2), dtype=int)
    for datapoint in datapoints:

        flag = np.argmin([distance(centroid['centroid'], datapoint) for centroid in clusters])
        clusters[flag]['datapoints'] = np.append(clusters[flag]['datapoints'], [datapoint], axis=0)

    return clusters

def printClusters(clusters):
    for i, cluster in enumerate(clusters):
        print(f"Cluster {i + 1}:")
        print(f"  Centroid: {cluster['centroid']}")
        print(f"  Datapoints:")
        for datapoint in cluster['datapoints']:
            print(f"    {datapoint}")
        print(f"  Label: {cluster['label']}\n")

def updateClusters(datapoints, clusters):
    for cluster in clusters:
        cluster['centroid'] = np.mean(cluster['datapoints'], axis=0)
    # cluster['centroid'] = np.round(np.mean(cluster['datapoints'], axis=0))
    datapoints_assign_clusters(datapoints, clusters)
    return clusters
#

def drawClusters(clusters, name):

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_title(name)
    colors = plt.get_cmap('tab10', len(clusters))

    for idx, cluster in enumerate(clusters):
        if cluster['datapoints'].size > 0:
            ax.scatter(cluster['datapoints'][:, 0], cluster['datapoints'][:, 1], color=colors(idx), label=f'Cluster {idx + 1}')
            ax.scatter(cluster['centroid'][0], cluster['centroid'][1], marker='*', color=colors(idx), s=200, edgecolor='white')

    ax.legend()
    plt.show()

if __name__ == '__main__':

    num_points = 100
    int_datapoints = create_N_datapoints(num_points)
    print(int_datapoints)

    K_centroids = 3
    clusters = create_K_clusters(int_datapoints, K_centroids)


    clusters = datapoints_assign_clusters(int_datapoints, clusters)
    i =0
    while True:

        i+=1
        drawClusters(clusters, f'K-Means Clustering' )
        printClusters(clusters)
        if clusters == updateClusters(int_datapoints, clusters):
            break
        clusters = updateClusters(int_datapoints, clusters)


