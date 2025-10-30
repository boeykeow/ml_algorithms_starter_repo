import numpy as np
def gini(y):
    if len(y)==0: return 0.0
    _, c = np.unique(y, return_counts=True); p=c/c.sum(); return 1-np.sum(p**2)
class Node:
    def __init__(self, depth=0, max_depth=5): self.depth=depth; self.max_depth=max_depth; self.left=self.right=None; self.feature=None; self.threshold=None; self.pred=None
class DecisionTree:
    def __init__(self, max_depth=5, min_samples_split=2): self.max_depth=max_depth; self.min_samples_split=min_samples_split; self.root=None
    def fit(self,X,y): self.root=self._build(X,y,0); return self
    def _build(self,X,y,depth):
        n=Node(depth,max_depth=self.max_depth)
        if depth>=self.max_depth or len(np.unique(y))==1 or len(y)<self.min_samples_split:
            n.pred = np.bincount(y).argmax(); return n
        best_feat,best_thr,best_gain=None,None,-1; cur=gini(y)
        for f in range(X.shape[1]):
            for thr in np.unique(X[:,f]):
                left = X[:,f] <= thr; right = ~left
                if left.sum()==0 or right.sum()==0: continue
                imp = (left.sum()/len(y))*gini(y[left]) + (right.sum()/len(y))*gini(y[right])
                gain = cur - imp
                if gain > best_gain:
                    best_gain = gain
                    best_feat, best_thr = f, thr
        if best_feat is None:
            n.pred = np.bincount(y).argmax(); return n
        n.feature, n.threshold = best_feat, best_thr
        left = X[:,best_feat] <= best_thr; right = ~left
        n.left = self._build(X[left], y[left], depth+1); n.right = self._build(X[right], y[right], depth+1); return n
    def _pred_row(self, n, x):
        if n.pred is not None: return n.pred
        return self._pred_row(n.left, x) if x[n.feature] <= n.threshold else self._pred_row(n.right, x)
    def predict(self,X): return np.array([self._pred_row(self.root, xi) for xi in X])
