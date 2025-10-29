import numpy as np
from .utils import add_bias
class LinearRegression:
    def __init__(self): self.w=None
    def fit(self, X, y):
        Xb = add_bias(X); self.w = np.linalg.pinv(Xb.T@Xb) @ (Xb.T@y); return self
    def predict(self, X):
        from .utils import add_bias; return add_bias(X) @ self.w
