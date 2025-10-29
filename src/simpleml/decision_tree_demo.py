from .datasets import blobs
from .decision_tree import DecisionTree
from .utils import train_test_split, accuracy
def main():
    X,y=blobs(n_per_class=120, centers=((0,0),(2,2)), spread=0.6, seed=2)
    Xtr,Xte,ytr,yte=train_test_split(X,y,seed=2)
    tree=DecisionTree(max_depth=4).fit(Xtr,ytr)
    print("Test acc:", round(accuracy(yte, tree.predict(Xte)),3))
if __name__=='__main__': main()
