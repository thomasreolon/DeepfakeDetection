import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from videoanalizer import VideoAnalizer
from videoanalizer.covariance import ALL_LABELS

# 1 get dataset
def get_dataset(vd, real_dir, fake_dir, rich=False):
    config = {'frames_per_sample':1000}
    # Real features
    X_r = vd.process_video(fdir=real_dir, config=config, rich=rich)
    if len(X_r)<100:
        config = {'frames_per_sample':max(10*len(X_r),300)}
        X_r = vd.process_video(fdir=real_dir, config=config, rich=rich)
    
    # Fake Features
    X_f = vd.process_video(fdir=fake_dir, config=config, rich=rich)
    if (len(X_r)>2*len(X_f)):
        config = {'frames_per_sample':max(10*len(X_f),250)}
        X_r = vd.process_video(fdir=fake_dir, config=config, rich=rich)

    # Dataset
    y = [1 for x in X_r] + [-1 for x in X_f]
    return (X_r+X_f, y)


# 2 -> features selector
def feature_selector(X,y, sel_type=0):
    x_r, x_f = [],[]
    # Split Data
    for a,b in zip(X,y):
        if b==1:
            x_r.append(a)
        else:
            x_f.append(a)
    # Get Means
    m_r = np.average(x_r, axis=0)
    m_f = np.average(x_f, axis=0)

    good_features, tot = set(), len(X[0])
    for _ in range(20):
        to_analize = [i for i in range(tot) if i not in good_features]
        best, best_score = -1, -1
        for i in to_analize:
            # mean distance
            centroid_r = [m_r[i]]
            centroid_f = [m_f[i]]
            for j in good_features:
                centroid_r.append(m_r[j])
                centroid_f.append(m_f[j])
            w = np.array(centroid_r)-centroid_f
            norm = np.linalg.norm(w)
            if norm>0:
                w /= norm

            w0 = -np.dot(w,(np.array(centroid_r)-centroid_f)/2)

            # variances
            v1 = []
            for a in x_r:
                b = [a[i]]
                for j in good_features:
                    b.append(a[j])
                v1.append(np.dot(w,b)+w0)
            v1 = (np.std(v1))**2
            v2 = []
            for a in x_f:
                b = [a[i]]
                for j in good_features:
                    b.append(a[j])
                v2.append(np.dot(w,b)+w0)
            v2 = (np.std(v2))**2
            score = norm/(v1+v2)
            if score>best_score:
                best_score=score
                best = i
        good_features.add(best)


    # use pca/lda to get some other insights
    if sel_type==1:
        pca = PCA(n_components=3)
        X_2 = pca.fit_transform(X)
        lda = LinearDiscriminantAnalysis()
        X_3 = lda.fit_transform(X, y)

    ## Select only features with best scores
    new_X = []
    for j,x in enumerate(X):
        tmp = []
        for i in good_features:
            tmp.append(x[i])
        if sel_type==1:
            tmp += list(X_2[j]) + list(X_3[j])
        new_X.append(tmp)

    return new_X, y, good_features
        
REAL='../test_data/videos/real/Obama'
FAKE='../test_data/videos/fake/Obama'
vd = VideoAnalizer()

X,Y = get_dataset(vd, REAL, FAKE, rich=True)
X,Y, gf = feature_selector(X,Y,sel_type=1)



exit(0)

# 3  general class for classifiers

class CLF():
    def __init__(self):
        pass
    def fit(self, X, Y):
        pass
    def predict(self, X, Y):
        pass

# 4.1 class for OneClassSVM


# 4.2 class for boosted


# 4.3 OneClassSVM only on some features


# 4.4 SVM only on some features


# 4.5 class boosted only on some features


# 5 train

# 6 compare presision results in a table





