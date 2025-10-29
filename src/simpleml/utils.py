import numpy as np
def train_test_split(X, y, test_size=0.3, seed=42):
    rng = np.random.default_rng(seed); idx = np.arange(len(X)); rng.shuffle(idx)
    s = int(len(X)*(1-test_size)); tr, te = idx[:s], idx[s:]; return X[tr], X[te], y[tr], y[te]
def accuracy(y_true, y_pred): return (y_true==y_pred).mean()
def rmse(y_true, y_pred): import numpy as np; return float(np.sqrt(np.mean((y_true-y_pred)**2)))
def add_bias(X): import numpy as np; return np.hstack([np.ones((X.shape[0],1)), X])
