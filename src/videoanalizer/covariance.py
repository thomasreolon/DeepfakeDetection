import numpy as np

def clean_features(features):
    data = []
    for _, values in features.items():
        data.append(values)
    return np.array(data)

def get_covariance(features):
    features = clean_features(features)
    return None