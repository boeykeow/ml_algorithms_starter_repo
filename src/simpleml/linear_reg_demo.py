from .datasets import linear_regression_data
from .linear_regression import LinearRegression
from .utils import rmse
def main():
    X,y=linear_regression_data(n=200, noise=0.6, seed=42)
    lr=LinearRegression().fit(X,y); preds=lr.predict(X)
    print("Weights:", lr.w); print("RMSE:", rmse(y,preds))
if __name__=='__main__': main()
