import numpy as np, json, sys, os, pathlib, random
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import OneClassSVM
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from videoanalizer import VideoAnalizer
from videoanalizer.covariance import ALL_LABELS
from prettytable import PrettyTable

os.chdir(pathlib.Path(__file__).parent.absolute()) # for debugging

# 1 get dataset
def get_dataset(vd, real_dir, fake_dir, rich=False, fps=1000, train_fraction=0.66, rich_features=0):
    config = {'frames_per_sample':fps}
    # Real features
    X_r, vids = vd.process_video(fdir=real_dir, config=config, rich_features=rich_features)
    if len(X_r)<100:
        config = {'frames_per_sample':max(int(fps/100*len(X_r)),300)}
        X_r, vids = vd.process_video(fdir=real_dir, config=config, rich_features=rich_features)
    X_r_train, X_r_test, labels_r = vd.split_train_test(X_r, vids, deterministic = True)

    # Fake Features
    X_f, vids = vd.process_video(fdir=fake_dir, config=config, rich_features=rich_features)
    if (len(X_r)>2*len(X_f)):
        config = {'frames_per_sample':max(int(fps/100*len(X_f)),250)}
        X_r, vids = vd.process_video(fdir=fake_dir, config=config, rich_features=rich_features)
    X_f_train, X_f_test, labels_f = vd.split_train_test(X_r, vids, labels_offset=(len(X_r_train), len(X_r_test)), deterministic = True)

    # Dataset
    x_train = X_r_train + X_f_train
    x_test  = X_r_test  + X_f_test
    y_train = [1 for x in X_r_train] + [-1 for x in X_f_train]
    y_test  = [1 for x in X_r_test] + [-1 for x in X_f_test]
    pt1={**labels_r['train'], **labels_f['train']}
    pt2={**labels_r['test'],  **labels_f['test']}
    labels  = {'train':pt1, 'test':pt2}
    return x_train, y_train, x_test, y_test, labels


# 2 -> features selector
SELECTOR_CACHE={}
def feature_selector(X,y, sel_type=0, override=False, pca_components = 3):
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
        SELECTOR_CACHE['good'] = good_features
        if sel_type == 2:
            pca = PCA(n_components=pca_components)
            SELECTOR_CACHE['pca'] = pca
        elif sel_type == 3:
            lda = LinearDiscriminantAnalysis()
            SELECTOR_CACHE['lda'] = lda
    else:
        good_features=SELECTOR_CACHE['good']
        if sel_type == 2:
            pca = SELECTOR_CACHE['pca']
        elif sel_type == 3:
            lda = SELECTOR_CACHE['lda']

    if sel_type == 1:
        new_X = []
        for j,x in enumerate(X):
            tmp = []
            for i in good_features:
                tmp.append(x[i])
            new_X.append(tmp)
    elif sel_type == 2:
        X_pca = pca.fit_transform(X)
        new_X = []
        for j,x in enumerate(X):
            new_X.append(list(X_pca[j])) #pca
    elif sel_type:
        X_lda = lda.fit_transform(X, y)
        new_X = []
        for j,x in enumerate(X):
            new_X.append(list(X_lda[j])) #lda

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

# 4.1 class for OneClassSVM
class OneClassRbf(CLF):
    def __init__(self):
        super().__init__('OneClassSVM_rbf')
        self.clf = OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
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
PATH = '../test_data/videos/{}/{}'
ENDC, OKCYAN, OKGREEN = '\033[0m', '\033[96m', '\033[92m'
OUT_DIR = '../output/results/'
DATASET = sys.argv[1]

if(not os.path.exists(OUT_DIR)):
    os.mkdir(OUT_DIR)

file = open(f"{OUT_DIR}result.tsv", "w+")

vd = VideoAnalizer()
what_features_are_selected = {}
avg_model_precision = {}
wrongly_classified = {}
best3_models = [(0,'None', 'None', 'None'), (0,'None', 'None', 'None'), (0,'None', 'None', 'None')]

for path in [DATASET]:    # for different people
    REAL_PATH = PATH.format('real', path)
    FAKE_PATH = PATH.format('fake', path)
    for rich in (0,1,2):      # with 190 features, with 250 features, with only rich_features
        x_train, y_train, x_test, y_test, labels = get_dataset(vd, REAL_PATH, FAKE_PATH, rich_features=rich)

        file.write("\n\n-------------------++SIZES++-------------------\n")
        file.write(f"   Training set size: {len(x_train)}\n")
        file.write(f"   Test set size: {len(x_test)}\n")
        file.write("-------------------++SIZES++-------------------\n\n")
        R = "190" if rich==0 else "250" if rich==1 else "only rich features"
        # 0 full features, 1 subset, 2 pca, 3 lda
        for selector in (0,1,2,3):
            x_train, y_train, gf = feature_selector(x_train, y_train,sel_type=selector, override=True)
            x_test, y_test, gf = feature_selector(x_test, y_test,sel_type=selector)
            # for f_id in gf:
            #     if f_id in what_features_are_selected:
            #         what_features_are_selected[f_id] += 1
            #     else:
            #         what_features_are_selected[f_id] = 1
            F = "all" if selector==0 else "best features" if selector==1 else "PCA" if selector==2 else "LDA"
            file.write(f"                      Rich:{R} | Features: {F}\n")
            table = PrettyTable(["Model", "Accuracy", "Confusion Matrix"])
            for Clf in (CLFPaper, CLFBoost, CLFLinear, CLFSVM, OneClassRbf): #for different models
                clf = Clf()
                clf.fit(x_train, y_train)
                y_pred = clf.predict(x_test)

                c_mat = confusion_matrix(y_test, y_pred, labels=[1,-1])

                # update models with best persormance
                avg_precision = (c_mat[0][0]/(c_mat[0][0]+c_mat[0][1])+c_mat[1][1]/(c_mat[1][1]+c_mat[1][0]))/2
                if best3_models[0][0] < avg_precision:
                    best3_models[0]=(avg_precision, clf.name, R, F)
                    for i in (1,2):
                        if best3_models[i][0] < best3_models[0][0]:
                            tmp             = best3_models[i]
                            best3_models[i] = best3_models[0]
                            best3_models[0] = tmp


                table.align["City name"] = "l"
                table.padding_width = 1
                table.add_row([f"{clf.name}",avg_precision, c_mat])
                # file.write(f"\nMODEL: \n")
                # file.write(f'Confusion matrix{c_mat}\n\n')
                # file.write(f'Average precision {avg_precision}\n<>-------------------------<>\n')
                print(f'\n{OKCYAN}rich:{rich},selector:{selector},model:{clf.name}{ENDC}')
                print(c_mat)

                # update avg. conf matrix
                if clf.name in avg_model_precision:
                    avg_model_precision[clf.name] += c_mat
                else:
                    avg_model_precision[clf.name] = np.array(c_mat)

                # which are wrongly classified
                # tot = {}
                # for yt, yp, i in zip(y_test, y_pred, range(9999)):
                #     file_name=f'{(yt==1 and "real") or "fake"}/{labels["test"][i]}'
                #     if not file_name in tot:
                #         tot[file_name] = [0,1]
                #     else:
                #         tot[file_name][1] += 1
                #     if yt != yp:
                #             tot[file_name][0] += 1
                # tmp = [(f,c[0]/c[1]) for f,c in tot.items()]
                # tmp.sort(key=lambda c: -c[1])                  # sort by % errors
                # file.write(f'errors:{tmp}\n')
                # file.write(f'_________________________\n')
                pass
            file.write(str(table))
            file.write("\n\n----------------------------------------------------------------------\n\n")
# print(f'{OKGREEN}WHAT FEATURES ARE SELECTED:{ENDC}')
# file.write(f'WHAT FEATURES ARE SELECTED:')
# wfs = list(what_features_are_selected.items())
# wfs.sort(reverse=True, key=lambda k: k[1])
# for f_id, count in wfs:
#     print(f'  {ALL_LABELS[f_id]}({f_id}): {count}')
#     file.write(f'  {ALL_LABELS[f_id]}({f_id}): {count}')

print(f'\n\n{OKGREEN}SUM OF CONF. MATRICES FOR EACH MODEL:{ENDC}')
file.write(f'\n\nSUM OF CONF. MATRICES FOR EACH MODEL:')
tmp = {k:[str(v[0]), str(v[1])] for k,v in avg_model_precision.items()}
print(json.dumps(tmp, indent=2))
file.write(f'{json.dumps(tmp, indent=2)}')

print(f'\n\n{OKGREEN}THE MODELS THAT HAD THE BEST AVG. PRECISION:{ENDC}')
file.write(f'\n\nTHE MODELS THAT HAD THE BEST AVG. PRECISION:')
print(json.dumps(best3_models, indent=2))
file.write(f'{json.dumps(best3_models, indent=2)}')

tmp = [(f,c[0]/c[1]) for f,c in wrongly_classified.items()]
tmp.sort(key=lambda c: -c[1])
tmp = [f'{f}         : {c}' for f,c in tmp]
print(f'\n\n{OKGREEN}FILES & MISSCLASSIFICATION PROBABILITY:{ENDC}')
file.write(f'\n\nFILES & MISSCLASSIFICATION PROBABILITY:')
print(json.dumps(tmp, indent=2))
file.write(f'{json.dumps(tmp, indent=2)}')

file.close()
