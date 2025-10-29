import numpy as np
from .decision_tree import DecisionTree
class RandomForest:
    def __init__(self, n_trees=5, max_depth=5, sample_ratio=0.8, feature_ratio=0.8, seed=0):
        self.n_trees=n_trees; self.max_depth=max_depth; self.sample_ratio=sample_ratio; self.feature_ratio=feature_ratio; self.seed=seed
        self.trees=[]; self.features_idx=[]
    def fit(self,X,y):
        rng=np.random.default_rng(self.seed); n,m=X.shape
        self.trees=[]; self.features_idx=[]
        for _ in range(self.n_trees):
            idx=rng.choice(n, size=int(n*self.sample_ratio), replace=True)
            feats=rng.choice(m, size=max(1,int(m*self.feature_ratio)), replace=False)
            tree=DecisionTree(max_depth=self.max_depth).fit(X[idx][:,feats], y[idx])
            self.trees.append(tree); self.features_idx.append(feats)
        return self
    def predict(self,X):
        votes=[t.predict(X[:,f]) for t,f in zip(self.trees,self.features_idx)]
        votes=np.stack(votes,axis=1)
        out=[]
        for row in votes:
            vals,counts=np.unique(row, return_counts=True); out.append(vals[counts.argmax()])
        return np.array(out)
