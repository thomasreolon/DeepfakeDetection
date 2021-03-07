from sklearn import svm
import joblib
import pathlib

class OneClassRbf():
    def __init__(self, video_analizer, rich):
        self.video_analizer = video_analizer
        self.clf = svm.OneClassSVM(nu=0.2, kernel="rbf", gamma='auto')
        self.rich = rich

    def set_video_analizer(self, video_analizer):
        """save association between classifier output and real class behind (eg. 1->Real, -1->Fake)"""
        self.video_analizer = video_analizer

    def get_sklearn_clf(self):
        return self.clf

    def predict_video(self, path_to_video, return_label=False):
        """
        input:
            - video path
        oputput:
            - integer that maps 'labels_map' into person is real or not
        """
        config = self.video_analizer._get_config({'interval':[(0,1e10)], 'frames_per_sample':-1})
        x, _ = self.video_analizer.process_video(files=[path_to_video], config=config, rich_features=self.rich)
        if(len(x) == 0):
            print(path_to_video)
        y = self.predict(x)[0]

        if (return_label): y = (y==1 and 'real' or 'fake')
        return y

    def fit(self, x):
        self.clf.fit(x)
        return self

    def predict(self, x):
        return self.clf.predict(x)


class BoostedOneClassRbf():
    def __init__(self, video_analizer, rich):
        self.video_analizer = video_analizer
        self.clf1 = svm.OneClassSVM(nu=0.2, kernel="rbf", gamma='auto')
        self.clf2 = svm.OneClassSVM(gamma='auto')
        f_path = pathlib.Path(__file__).parent.absolute().joinpath(f'pretrainedSVC_{rich}.joblib')
        self.clf3 = joblib.load(f_path)
        self.rich = rich

    def set_video_analizer(self, video_analizer):
        """save association between classifier output and real class behind (eg. 1->Real, -1->Fake)"""
        self.video_analizer = video_analizer

    def get_sklearn_clf(self):
        return self.clf1

    def predict_video(self, path_to_video, return_label=False):
        """
        input:
            - video path
        oputput:
            - integer that maps 'labels_map' into person is real or not
        """
        config = self.video_analizer._get_config({'interval':[(0,1e10)], 'frames_per_sample':-1})
        x, _ = self.video_analizer.process_video(files=[path_to_video], config=config, rich_features=self.rich)
        y = self.predict(x)[0]

        if (return_label): y = (y==1 and 'real' or 'fake')
        return y

    def fit(self, x):
        self.clf1.fit(x)
        self.clf2.fit(x)
        return self

    def predict(self, x):
        res=[]
        y1= self.clf1.predict(x)
        y2= self.clf2.predict(x)
        if(self.rich == 0):
            y3= self.clf3.predict([x[0][:190]])
        elif(self.rich == 1):
            y3= self.clf3.predict([x[0][:250]])
        else:
            y3= self.clf3.predict([x[0][:60]])

        for r1,r2,r3 in zip(y1,y2,y3):
            res.append((r1+r2+r3>0 and 1) or -1)
        return res
