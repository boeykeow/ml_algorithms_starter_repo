from .datasets import xor_data
from .nn import TinyMLP
from .utils import train_test_split, accuracy
def main():
    X,y=xor_data(n=200, spread=0.3, seed=5)
    Xtr,Xte,ytr,yte=train_test_split(X,y,seed=5)
    nn=TinyMLP(hidden=8, lr=0.1, epochs=3000, seed=5).fit(Xtr,ytr)
    print("TinyMLP acc:", round(accuracy(yte, nn.predict(Xte)),3))
if __name__=='__main__': main()
