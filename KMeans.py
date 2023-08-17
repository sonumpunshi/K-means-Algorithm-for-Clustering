import numpy as np
import matplotlib.pyplot as plt
import random
import sys


def read_dataset(filename):
    data = np.loadtxt(filename, delimiter=',')
    return data[:, :2], data[:, 2]


def initialize_centroids(X, K):
    centroids = X[np.random.choice(X.shape[0], K, replace=False)]
    return centroids


def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2, axis=1))


def visualize_clusters(X, centroids, labels, K, iteration):
    plt.figure()
    for k in range(K):
        cluster_points = X[labels == k]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {k + 1}')
        plt.scatter(centroids[k, 0], centroids[k, 1], marker='x', color='k', s=100)
    plt.title(f'K = {K}, Iteration: {iteration}')
    plt.legend()

    # Save the image to disk
    plt.savefig(f'K_{K}_Iteration_{iteration}.png')


def KMeans(datasetFile, K=2):
    X, y = read_dataset(datasetFile)
    centroids = initialize_centroids(X, K)
    prev_centroids = np.zeros(centroids.shape)
    labels = np.zeros(X.shape[0])

    iteration = 0
    converged = False
    while not converged:
        iteration += 1
        if iteration <= 3 or iteration % 5 == 0 or np.array_equal(centroids, prev_centroids):
            visualize_clusters(X, centroids, labels, K, iteration)

        for i, x in enumerate(X):
            distances = euclidean_distance(x, centroids)
            labels[i] = np.argmin(distances)

        prev_centroids = np.copy(centroids)
        for k in range(K):
            cluster_points = X[labels == k]
            centroids[k] = np.mean(cluster_points, axis=0)

        converged = np.array_equal(centroids, prev_centroids)


if __name__ == "__main__":
    datasetFile = 'ClusteringData.txt'
    KMeans(datasetFile)
    KMeans(datasetFile, K=3)
    KMeans(datasetFile, K=4)
