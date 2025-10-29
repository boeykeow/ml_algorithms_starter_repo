import numpy as np
def blobs(n_per_class=50, centers=((0,0),(2,2)), spread=0.5, seed=0):
    rng = np.random.default_rng(seed); Xl, yl = [], []
    for i,(cx,cy) in enumerate(centers):
        Xc = rng.normal((cx,cy), spread, size=(n_per_class,2)); yc = np.full(n_per_class, i, dtype=int)
        Xl.append(Xc); yl.append(yc)
    return np.vstack(Xl), np.concatenate(yl)
def linear_regression_data(n=200, noise=0.5, seed=0):
    rng = np.random.default_rng(seed); X = rng.uniform(-3,3,size=(n,1))
    y = 2.0*X[:,0]-1.0 + rng.normal(0,noise,size=n); return X,y
def xor_data(n=200, spread=0.3, seed=0):
    rng = np.random.default_rng(seed); centers=[(-1,-1),(1,1),(-1,1),(1,-1)]; labels=[0,0,1,1]
    Xl, yl = [], []
    for (cx,cy),lab in zip(centers,labels):
        Xc = rng.normal((cx,cy), spread, size=(n//4,2)); yc = np.full(n//4, lab, dtype=int)
        Xl.append(Xc); yl.append(yc)
    return np.vstack(Xl), np.concatenate(yl)
