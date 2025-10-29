import numpy as np
class KMeans:
    def __init__(self, k=2, max_iter=100, seed=0): self.k=k; self.max_iter=max_iter; self.seed=seed; self.centers=None
    def fit(self, X):
        rng=np.random.default_rng(self.seed); centers=X[rng.choice(len(X), size=self.k, replace=False)].copy()
        for _ in range(self.max_iter):
            d=np.linalg.norm(X[:,None,:]-centers[None,:,:], axis=2); labels=d.argmin(axis=1)
            new=np.array([X[labels==j].mean(axis=0) if np.any(labels==j) else centers[j] for j in range(self.k)])
            if np.allclose(new,centers): break
            centers=new
        self.centers=centers; self.labels_=labels; return self
    def predict(self,X):
        d=np.linalg.norm(X[:,None,:]-self.centers[None,:,:], axis=2); return d.argmin(axis=1)
