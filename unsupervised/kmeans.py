import numpy as np
import matplotlib.pyplot as plt


def init_centroids(X, k):
    """
    Args:
        X (ndarray): m, n
        k (int)
    """

    m = X.shape[0]
    shuffle_ids = np.random.permutation(m)
    return X[shuffle_ids[:k]]


def assign_centroids(X, centroids):
    """
    Args:
        X (ndarray): m, n
        centroids (ndarray): k, n
    """
    
    X_exp = np.expand_dims(X, axis=(1,))
    centroids_exp = np.expand_dims(centroids, axis=(0,))

    # X_exp - centroids_exp -> (m, k, n)
    l2_norm =  np.sqrt(((X_exp - centroids_exp)**2).sum(axis=-1)) # (m, k)

    return np.argmin(l2_norm, axis=-1) # m


def compute_centroids(X, ids, k):
    n = X.shape[1]
    centroids = np.zeros((k, n))
    for id in range(k):
        centroids[id] = X[(ids==id)].mean(axis=0)
    return centroids


def _k_means(X, k, n_iters):
    centroids = init_centroids(X, k)
    for iter in range(n_iters):
        # assign labels to samples
        ids = assign_centroids(X, centroids)
        # refine centroids
        centroids = compute_centroids(X, ids, k)
    
    return centroids, ids


def compute_cost(X, ids, centroids, k):
    cost = 0
    for id in range(k):
        cost += ((X[(ids==id)] - centroids[id])**2).mean()
    return cost / k


def k_means(X, n_init: int, k: int, n_iters: int):

    min_cost = float("inf")
    best_centroids = None
    best_ids = None
    for _ in range(n_init):
        centroids, ids = _k_means(X, k, n_iters)
        cost = compute_cost(X, ids, centroids, k)
        if cost < min_cost:
            min_cost = cost
            best_centroids = centroids
            best_ids = ids

    return best_centroids, best_ids


def plot_data(X, ids, centroids):
    plt.clf()
    plt.scatter(X[:, 0], X[:, 1], c=ids, marker='o')
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='x')
    plt.show()

if __name__ == "__main__":

    X = np.load(open("unsupervised/X_dummy.npy", "rb"))
    centroids, ids = k_means(X, n_init=5, k=3, n_iters=10)
    plot_data(X, ids, centroids)