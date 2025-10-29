import numpy as np
from .datasets import blobs
from .svm import LinearSVM
from .utils import train_test_split, accuracy
def main():
    X,y=blobs(n_per_class=120, centers=((0,0),(2,2)), spread=0.7, seed=4)
    y=np.where(y==0,-1,1)
    Xtr,Xte,ytr,yte=train_test_split(X,y,seed=4)
    svm=LinearSVM(lr=0.05, epochs=2000, C=1.0).fit(Xtr,ytr)
    pred=svm.predict(Xte)
    print("Linear SVM acc:", round((pred==yte).mean(),3))
if __name__=='__main__': main()
