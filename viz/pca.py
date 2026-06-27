import numpy as np
import matplotlib.pyplot as plt


class PCA:
    def __init__(self, k):
        self.k = k
        self.mean = None
        self.std = None
        self.princ_components = None

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        Xn = (X - self.mean) / self.std
        cov = (Xn.T @ Xn) / (len(X) - 1)
        vals, vecs = np.linalg.eigh(cov)
        idx = np.argsort(vals)[::-1]
        self.princ_components = vecs[:, idx][:, :self.k] # first k components with highest eigen values
        self.explained_variance_ratio_ = \
            np.sort(vals)[::-1][:self.k] / vals.sum()
        return self

    def transform(self, X):
        return ((X - self.mean) / self.std) @ self.princ_components

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def inverse_transform(self, X_proj):
        return X_proj @ self.princ_components.T * self.std + self.mean



X = np.array([[2.5, 2.4],
              [0.5, 0.7],
              [2.2, 2.9],
              [1.9, 2.2],
              [3.1, 3.0]])

pca = PCA(k=1)
X_low = pca.fit_transform(X)
X_back = pca.inverse_transform(X_low)

print(pca.explained_variance_ratio_)
x1, x2 = zip(*((X - pca.mean)/pca.std))

pc1 = pca.princ_components[:, 0]

pc1_l_start = X_low.min() * pc1
pc1_l_end = X_low.max() * pc1

pc1_xs, pc1_ys = zip(*(pc1_l_start, pc1_l_end))

plt.subplot(1,2,1).scatter(x1, x2, label='Centered original')
plt.subplot(1,2,1).plot(pc1_xs, pc1_ys, label='PC1')
plt.subplot(1,2,2).scatter(X_low, np.zeros_like(X_low))
plt.subplot(1,2,1).legend()
plt.show()

