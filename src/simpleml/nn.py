import numpy as np
class TinyMLP:
    def __init__(self, hidden=8, lr=0.1, epochs=2000, seed=0): self.hidden=hidden; self.lr=lr; self.epochs=epochs; self.seed=seed
    def fit(self,X,y):
        rng=np.random.default_rng(self.seed); n,d=X.shape; y=y.reshape(-1,1)
        W1=rng.normal(0,0.1,size=(d,self.hidden)); b1=np.zeros((1,self.hidden))
        W2=rng.normal(0,0.1,size=(self.hidden,1)); b2=np.zeros((1,1))
        for _ in range(self.epochs):
            Z1 = X @ W1 + b1
            A1 = np.tanh(Z1)
            Z2 = A1 @ W2 + b2
            A2 = 1/(1+np.exp(-Z2))
            dZ2 = A2 - y
            dW2 = (A1.T @ dZ2) / n
            db2 = dZ2.mean(axis=0, keepdims=True)
            dA1 = dZ2 @ W2.T
            dZ1 = dA1 * (1 - np.tanh(Z1)**2)
            dW1 = (X.T @ dZ1) / n
            db1 = dZ1.mean(axis=0, keepdims=True)
            W1 -= self.lr * dW1
            b1 -= self.lr * db1
            W2 -= self.lr * dW2
            b2 -= self.lr * db2
        # store trained parameters
        self.W1 = W1
        self.b1 = b1
        self.W2 = W2
        self.b2 = b2
        return self
    def predict_proba(self,X):
        A1=np.tanh(X@self.W1+self.b1); A2=1/(1+np.exp(-(A1@self.W2+self.b2))); return A2.ravel()
    def predict(self,X): return (self.predict_proba(X)>=0.5).astype(int)
