import numpy as np, math

def pearson_correlation(X, Y):
    covariance = np.cov(X,Y)[0][1]
    X_std = np.std(X)
    Y_std = np.std(Y)
    if((X_std == 0) or (Y_std == 0)): #to check if it's correct (for now just to avoid /0)
        pc = 0.0000
    else:
        pc = covariance/(X_std*Y_std)
    return round(pc, 4)
