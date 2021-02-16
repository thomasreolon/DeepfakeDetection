import numpy as np
import os, pathlib, random
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import OneClassSVM
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from videoanalizer import VideoAnalizer
from videoanalizer.covariance import ALL_LABELS
import json

os.chdir(pathlib.Path(__file__).parent.absolute()) # for debugging

# 1 get dataset
def get_dataset(vd, real_dir, fake_dir, rich=False, fps=1000):
    config = {'frames_per_sample':fps}
    # Real features
    X_r, vids = vd.process_video(fdir=real_dir, config=config, rich=rich)
    if len(X_r)<100:
        config = {'frames_per_sample':max(int(fps/100*len(X_r)),300)}
        X_r, vids = vd.process_video(fdir=real_dir, config=config, rich=rich)
    X_r_train, X_r_test = vd.split_train_test(X_r, vids)
    
    # Fake Features
    X_f, vids = vd.process_video(fdir=fake_dir, config=config, rich=rich)
    if (len(X_r)>2*len(X_f)):
        config = {'frames_per_sample':max(int(fps/100*len(X_f)),250)}
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
        if sel_type>1:
            pca = SELECTOR_CACHE['pca']
            lda = SELECTOR_CACHE['lda']
        good_features=SELECTOR_CACHE['good']
    if sel_type>1:
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
        onefake,onereal=False, False
        for x, yp2, yp, yt in zip(X, ypred2, ypred, y):
            if yp!=yp2 or random.random()>0.4:
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
PATH='../test_data/videos/{}/{}'
ENDC, OKCYAN, OKGREEN = '\033[0m', '\033[96m', '\033[92m'


vd = VideoAnalizer()
what_features_are_selected = {}
avg_model_precision = {}
best3_models = [(0,'None'), (0,'None'), (0,'None')]

for person in ['Obama']:    # for different people
    REAL_PATH = PATH.format('real', person)
    FAKE_PATH = PATH.format('fake', person)
    for iteration in (0,1,2):           # split dataset in 3 different ways
        for rich in (True, False):      # with 190 features, with 250 features
            x_train, y_train, x_test, y_test = get_dataset(vd, REAL_PATH, FAKE_PATH, rich=rich, fps=300+iteration*200)

            for selector in (0,1,2):    # train with full features, a subset, a subset+pca+lda
                x_train, y_train, gf = feature_selector(x_train, y_train,sel_type=selector, override=True)
                x_test, y_test, gf = feature_selector(x_test, y_test,sel_type=selector)
                for f_id in gf:
                    if f_id in what_features_are_selected:
                        what_features_are_selected[f_id] += 1
                    else:
                        what_features_are_selected[f_id] = 1

                for Clf in (CLFPaper, CLFBoost, CLFLinear, CLFSVM): #for different models
                    clf = Clf()
                    clf.fit(x_train, y_train)
                    y_pred = clf.predict(x_test)

                    c_mat = confusion_matrix(y_test, y_pred, labels=[1,-1])
                    print(f'\n{OKCYAN}iteration:{iteration},rich:{rich},selector:{selector},model:{clf.name}{ENDC}')
                    print(c_mat)
                    
                    # update models with best persormance
                    avg_precision = (c_mat[0][0]/(c_mat[0][0]+c_mat[0][1])+c_mat[1][1]/(c_mat[1][1]+c_mat[1][0]))/2
                    if best3_models[0][0] < avg_precision:
                        best3_models[0]=(avg_precision, clf.name)
                        for i in (1,2):
                            if best3_models[i][0] < best3_models[0][0]:
                                tmp             = best3_models[i]
                                best3_models[i] = best3_models[0]
                                best3_models[0] = tmp
                    
                    # update avg. conf matrix
                    if clf.name in avg_model_precision:
                        avg_model_precision[clf.name] += c_mat
                    else:
                        avg_model_precision[clf.name] = np.array(c_mat)

print(f'{OKGREEN}WHAT FEATURES ARE SELECTED:{ENDC}')
wfs = list(what_features_are_selected.items())
wfs.sort(reverse=True, key=lambda k: k[1])
for f_id, count in wfs:
    print(f'  {ALL_LABELS[f_id]}({f_id}): {count}')

print(f'\n{OKGREEN}SUM OF CONF. MATRICES FOR EACH MODEL:{ENDC}')
tmp = {k:[str(v[0]), str(v[1])] for k,v in avg_model_precision.items()}
print(json.dumps(tmp, indent=2))

print(f'\n{OKGREEN}THE MODELS THAT HAD THE BEST AVG. PRECISION:{ENDC}')
print(json.dumps(best3_models, indent=2))

"""
_____________________________________________________________________________________________




PREVIOUS EXPERIMENTS SHOWED GOOD RESULTS:
- using 250 features instead of 190
- using 250 features and reducing them with LDA, SFS, ...

rich:False, selector:0
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