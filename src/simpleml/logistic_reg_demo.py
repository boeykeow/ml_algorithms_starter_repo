from .datasets import blobs
from .logistic_regression import LogisticRegressionGD
from .utils import train_test_split, accuracy
def main():
    X,y=blobs(n_per_class=100, centers=((0,0),(2,2)), spread=0.7, seed=0)
    Xtr,Xte,ytr,yte=train_test_split(X,y,0.3,seed=0)
    clf=LogisticRegressionGD(lr=0.2, epochs=1500).fit(Xtr,ytr)
    print("Test acc:", round(accuracy(yte, clf.predict(Xte)),3))
if __name__=='__main__': main()
