import numpy as np

CORRELATION_LABELS=[]
ALL_LABELS=[]

def pearson_correlation(X, Y):
    covariance = np.cov(X,Y)[0][1]
    X_std = np.std(X)
    Y_std = np.std(Y)
    if((X_std == 0) or (Y_std == 0)):
        pc = 0.0000
    else:
        pc = covariance/(X_std*Y_std)
    return round(pc, 4)

def update_labels(features):
    #global CORRELATION_LABELS
    if len(CORRELATION_LABELS)==0:
        features = list(features.keys())
        for i in range(len(features)):
            for j in range(i+1, len(features)):
                CORRELATION_LABELS.append(f'{features[i]} π {features[j]}')


def get_correlation_matrix(features):
    update_labels(features)
    correlation_matrix = []

    for f0 in features:
        f0 = features[f0]
        row = []
        for f1 in features:
            f1 = features[f1]
            row.append(pearson_correlation(f0, f1))
        correlation_matrix.append(row)
    return correlation_matrix

def get_190_features(features):
    """
    do correlation between AUs and return unique values of the matrix
    """
    c_mat = get_correlation_matrix(features)
    features = []
    for i in range(len(c_mat)):
        for j in range(len(c_mat[0])):
            if (i<j):
                features.append(c_mat[i][j])
    return features

def get_rich_features(features):
    #global ALL_LABELS
    if len(ALL_LABELS)==0:
        update_labels(features)
        for f in CORRELATION_LABELS:
            ALL_LABELS.append(f)
        for f in features.keys():
            ALL_LABELS.append(f'avg.{f}')
            ALL_LABELS.append(f'std.{f}')
            ALL_LABELS.append(f'max {f}')

    X = []
    for _, val in features.items():
        if len(val)>0:
            ma  = np.max(val)
            av  = np.average(val)
            std = np.std(val)
        else:
            ma, av, std = 0., 0., 0.
        X += [av, std, ma]
    return X
