import numpy as np
import os, pathlib, random
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import OneClassSVM
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from videoanalizer import VideoAnalizer
from videoanalizer.covariance import ALL_LABELS

os.chdir(pathlib.Path(__file__).parent.absolute()) # for debugging

# 1 get dataset
def get_dataset(vd, real_dir, fake_dir, rich=False):
    config = {'frames_per_sample':1000}
    # Real features
    X_r, vids = vd.process_video(fdir=real_dir, config=config, rich=rich)
    if len(X_r)<100:
        config = {'frames_per_sample':max(10*len(X_r),300)}
        X_r, vids = vd.process_video(fdir=real_dir, config=config, rich=rich)
    X_r_train, X_r_test = vd.split_train_test(X_r, vids)
    
    # Fake Features
    X_f, vids = vd.process_video(fdir=fake_dir, config=config, rich=rich)
    if (len(X_r)>2*len(X_f)):
        config = {'frames_per_sample':max(10*len(X_f),250)}
        X_r, vids = vd.process_video(fdir=fake_dir, config=config, rich=rich)
    X_f_train, X_f_test = vd.split_train_test(X_r, vids)

    # Dataset
    x_train = X_r_train + X_f_train
    x_test  = X_r_test  + X_f_test
    y_train = [1 for x in X_r_train] + [-1 for x in X_f_train]
    y_test  = [1 for x in X_r_test] + [-1 for x in X_f_test]
    return x_train, y_train, x_test, y_test


# 2 -> features selector
SELECTOR_CACHE={}
def feature_selector(X,y, sel_type=0, override=False):
    if sel_type==0:
        return X,y,set()
    
    if override:
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
                score = norm/((v1+v2) or 1)
                if score>best_score:
                    best_score=score
                    best = i
            good_features.add(best)


        # use pca/lda to get some other insights
        if sel_type>1:
            pca = PCA(n_components=3)
            lda = LinearDiscriminantAnalysis()
            SELECTOR_CACHE['pca'] = pca
            SELECTOR_CACHE['lda'] = lda
        SELECTOR_CACHE['good'] = good_features
    else:
        pca = SELECTOR_CACHE['pca']
        lda = SELECTOR_CACHE['lda']
        good_features=SELECTOR_CACHE['good']

    X_2 = pca.fit_transform(X)
    X_3 = lda.fit_transform(X, y)

    ## Select only features with best scores
    new_X = []
    for j,x in enumerate(X):
        tmp = []
        for i in good_features:
            tmp.append(x[i])
        if sel_type>1:
            tmp += list(X_2[j]) + list(X_3[j])
        new_X.append(tmp)

    return new_X, y, good_features

# 3  general class for classifiers

class CLF():
    def __init__(self, name=None):
        self.name = name or 'base clf'
    def fit(self, X, y):
        pass
    def predict(self, X):
        pass

# 4.1 class for OneClassSVM
class CLFPaper(CLF):
    def __init__(self):
        super().__init__('OneClassSVM')
        self.clf = OneClassSVM(gamma='auto')
    def fit(self,X,y):
        D=[]
        for a,b in zip(X,y):
            if b==1 or random.random()>0.95:
                D.append(a)
        self.clf.fit(D)
    def predict(self, X):
        return self.clf.predict(X)
    

# 4.2 class for boosting
class CLFBoost(CLF):
    def __init__(self):
        super().__init__('Boosting')
        self.clf1 = SVC(kernel='linear')
        self.clf2 = OneClassSVM(gamma='auto', nu=0.2)
        self.clf3 = SVC(kernel='rbf', gamma='auto')
    def fit(self,X,y):
        # CLF1
        X1,y1=[],[]
        for a,b in zip(X,y):
            if random.random()>0.4:
                X1.append(a)
                y1.append(b)
        self.clf1.fit(X1,y1)
        ypred = self.clf1.predict(X)
        # CLF2
        X2=[]
        for x, yt, yp in zip(X, y, ypred):
            if yt==1 and (yt!=yp or random.random()>0.6):
                X2.append(x)
        self.clf2.fit(X2)
        ypred2 = self.clf2.predict(X)
        # CLF3
        X3,y3=[],[]
        for x, yp2, yp, yt in zip(X, ypred2, ypred, y):
            if yp!=yp2:
                X3.append(x)
                y3.append(yt)
        self.clf3.fit(X3, y3)
    def predict(self, X):
        res = []
        y1 = self.clf1.predict(X)
        y2 = self.clf2.predict(X)
        y3 = self.clf3.predict(X)
        for a,b,c in zip(y1,y2,y3):
            if a==b:
                res.append(a)
            else:
                res.append(c)
        return res

# 4.3 class just LDA
class CLFLinear(CLF):
    def __init__(self):
        super().__init__('LinearDiscriminantAnalysis')
        self.clf = LinearDiscriminantAnalysis()
    def fit(self,X,y):
        self.clf.fit(X,y)
    def predict(self, X):
        return self.clf.predict(X)
   
# 4.4 just SVM:rbf
class CLFSVM(CLF):
    def __init__(self):
        super().__init__('SVM-rbf')
        self.clf = SVC(kernel='rbf', gamma='auto')
    def fit(self,X,y):
        self.clf.fit(X,y)
    def predict(self, X):
        return self.clf.predict(X)
   

# INIT
REAL='../test_data/videos/real/Obama'
FAKE='../test_data/videos/fake/Obama'
vd = VideoAnalizer()
RICH=True
SEL_TYPE = 2

x_train, y_train, x_test, y_test = get_dataset(vd, REAL, FAKE, rich=RICH)
# Train
x_train, y_train, gf = feature_selector(x_train, y_train,sel_type=SEL_TYPE, override=True)

c1 = CLFPaper()
c1.fit(x_train, y_train)
c2 = CLFBoost()
c2.fit(x_train, y_train)


# Test

x_test, y_test, gf = feature_selector(x_test, y_test,sel_type=SEL_TYPE)

y_pred1 = c1.predict(x_test)
y_pred2 = c2.predict(x_test)

print('CONF MATRIX C1')
print(confusion_matrix(y_test, y_pred1))


print('CONF MATRIX C2')
print(confusion_matrix(y_test, y_pred2))



"""
_____________________________________________________________________________________________
False, 0

CONF MATRIX C1
[[5 9]
 [9 7]]
CONF MATRIX C2
[[ 4 10]
 [ 2 14]]
_____________________________________________________________________________________________
True, 0
CONF MATRIX C1
[[10  4]
 [ 5 11]]
CONF MATRIX C2
[[10  4]
 [ 3 13]]
_____________________________________________________________________________________________
False, 1
CONF MATRIX C1
[[ 3 11]
 [ 9  7]]
CONF MATRIX C2
[[ 0 14]
 [ 0 16]]
_____________________________________________________________________________________________
True, 1
CONF MATRIX C1
[[8 6]
 [9 7]]
CONF MATRIX C2
[[14  0]
 [15  1]]
_____________________________________________________________________________________________
False, 2
CONF MATRIX C1
[[14  0]
 [ 7  9]]
CONF MATRIX C2
[[13  1]
 [ 0 16]]
_____________________________________________________________________________________________
True, 2
CONF MATRIX C1
[[13  1]
 [ 7  9]]
CONF MATRIX C2
[[13  1]
 [ 0 16]]
_____________________________________________________________________________________________
"""