import numpy as np, math

def pearson_correlation(X, Y):
    X_mean = np.mean(X)
    Y_mean = np.mean(Y)
    covariance = np.cov(X,Y)[0][1]
    X_std = np.std(X)
    Y_std = np.std(Y)
    if((X_std == 0) or (Y_std == 0)):
        pc = 0.0000
    else:
        pc = round(covariance/(X_std*Y_std), 4)
    if(math.isnan(pc)):
        print(X)
        print(Y)
        quit()
    return pc
