from .datasets import blobs
from .kmeans import KMeans
def main():
    X,_=blobs(n_per_class=60, centers=((0,0),(3,3),(0,3)), spread=0.5, seed=1)
    km=KMeans(k=3, max_iter=100, seed=1).fit(X)
    print("Centers:", km.centers); print("Counts:", {i:(km.labels_==i).sum() for i in range(3)})
if __name__=='__main__': main()
