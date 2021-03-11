from sklearn import svm
import joblib
import pathlib
from sklearn.model_selection import GridSearchCV
from .video_edits import save_video_landmarks

class OneClassRbf():
    def __init__(self, video_analizer, rich, person=None, config=None, gridsearch=False):
        self.video_analizer = video_analizer
        self.gridsearch = gridsearch
        if(gridsearch):
            kernels = ['linear', 'rbf', 'poly', 'rbf', 'sigmoid']
            gammas = ['scale', 'auto']
            nu = [0.05,0.1,0.2,0.3,0.4,0.5]
            clfs = []
            for k in kernels:
                for g in gammas:
                    for n in nu:
                        clfs.append({"Classifier" : svm.OneClassSVM(kernel=k, gamma=g, nu=n), "Result" : None, "Description" : f"Kernel: {k}, gamma: {g}, nu: {n}"})
            self.clfs = clfs
        else:
            self.clf = svm.OneClassSVM(kernel='poly', gamma=0.001, degree=5, nu=0.4, shrinking=False)
        self.rich = rich
        self.config = config


    def set_video_analizer(self, video_analizer):
        """save association between classifier output and real class behind (eg. 1->Real, -1->Fake)"""
        self.video_analizer = video_analizer

    def get_sklearn_clf(self):
        return self.clf

    def predict_video(self, path_to_video, return_label=False, landmark_video=False):
        """
        input:
            - video path
        oputput:
            - integer that maps 'labels_map' into person is real or not
        """
        config = self.config or self.video_analizer._get_config({'interval':[(0,1e10)], 'frames_per_sample':300})
        x, _ = self.video_analizer.process_video(files=[path_to_video], config=config, rich_features=self.rich)
        if len(x)==0:      raise Exception(f'0 samples have been extracted from {path_to_video}')

        if(self.gridsearch):
            predictions = self.predict(x)
            results = []
            for i,y in enumerate(predictions):
                y = [max(0,v) for v in y]
                y = sum(y)/len(y)
                self.clfs[i]["Result"] = (y>0.5 and 'real' or 'fake')
            return self.clfs
        else:
            y = self.predict(x)
            y = [max(0,v) for v in y]
            y = sum(y)/len(y)
            label = (y>0.5 and 'real' or 'fake')

        if landmark_video:
            save_video_landmarks(path_to_video, label, self.video_analizer)

        return return_label and label or y

    def fit(self, x):
        if(self.gridsearch):
            for i in range(0, len(self.clfs)):
                self.clfs[i]["Classifier"].fit(x)
        else:
            self.clf.fit(x)
        return self

    def predict(self, x):
        if(self.gridsearch):
            predictions = []
            for i in range(0, len(self.clfs)):
                predictions.append(self.clfs[i]["Classifier"].predict(x))
            return predictions
        else:
            return self.clf.predict(x)


class BoostedOneClassRbf(OneClassRbf):
    def __init__(self, video_analizer, rich, person, config=None, gridsearch=False):
        self.video_analizer = video_analizer
        self.clf1 = svm.OneClassSVM(kernel='poly', gamma=0.001, degree=5, nu=0.4, shrinking=False)
        self.clf2 = svm.OneClassSVM(kernel='rbf', gamma='scale', degree=5, nu=0.01, shrinking=True)
        f_path = pathlib.Path(__file__).parent.absolute().joinpath(f'pretrainedSVC_{rich}_{person}.joblib')
        self.clf3 = joblib.load(f_path)
        self.rich = rich
        self.config = config

    def get_sklearn_clf(self):
        return self.clf1


    def fit(self, x):
        self.clf1.fit(x)
        self.clf2.fit(x)
        return self

    def predict(self, x):
        res=[]
        y1= self.clf1.predict(x)
        y2= self.clf2.predict(x)
        y3= self.clf3.predict(x)

        for r1,r2,r3 in zip(y1,y2,y3):
            res.append((r1+r2+r3>0 and 1) or -1)
        return res
