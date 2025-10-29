import numpy as np
from .utils import add_bias
class LogisticRegressionGD:
    def __init__(self, lr=0.1, epochs=1000): self.lr=lr; self.epochs=epochs; self.w=None
    def _sigmoid(self,z): return 1/(1+np.exp(-z))
    def fit(self,X,y):
        Xb=add_bias(X); self.w=np.zeros(Xb.shape[1])
        for _ in range(self.epochs):
            z=Xb@self.w; yhat=self._sigmoid(z); grad = Xb.T @ (yhat - y) / len(y); self.w -= self.lr*grad
        return self
    def predict(self,X): 
        Xb=add_bias(X); return (self._sigmoid(Xb@self.w) >= 0.5).astype(int)
