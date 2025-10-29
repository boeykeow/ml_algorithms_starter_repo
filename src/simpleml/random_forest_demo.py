from .datasets import blobs
from .random_forest import RandomForest
from .utils import train_test_split, accuracy
def main():
    X,y=blobs(n_per_class=150, centers=((0,0),(2,2)), spread=0.7, seed=3)
    Xtr,Xte,ytr,yte=train_test_split(X,y,seed=3)
    rf=RandomForest(n_trees=9, max_depth=5, seed=3).fit(Xtr,ytr)
    print("RF acc:", round(accuracy(yte, rf.predict(Xte)),3))
if __name__=='__main__': main()
