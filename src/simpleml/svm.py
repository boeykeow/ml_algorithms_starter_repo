import numpy as np
from .utils import add_bias
class LinearSVM:
    def __init__(self, lr=0.1, epochs=1000, C=1.0): self.lr=lr; self.epochs=epochs; self.C=C; self.w=None
    def fit(self,X,y):
        Xb=add_bias(X); self.w=np.zeros(Xb.shape[1])
        for _ in range(self.epochs):
            margins=y*(Xb@self.w); mis = margins<1; grad=self.w.copy()
            if np.any(mis):
                grad -= self.C * (Xb[mis] * y[mis,None]).sum(axis=0)
            self.w -= self.lr * grad / len(y)
        return self
    def predict(self,X):
        Xb=add_bias(X); score=Xb@self.w; return np.where(score>=0,1,-1)
