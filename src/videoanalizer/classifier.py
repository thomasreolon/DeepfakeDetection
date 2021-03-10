from sklearn import svm
import joblib
import pathlib
from .video_edits import save_video_landmarks

class OneClassRbf():
    def __init__(self, video_analizer, rich, person):
        self.video_analizer = video_analizer
        self.clf = svm.OneClassSVM(nu=0.2, kernel="rbf", gamma='auto')
        self.rich = rich

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
        config = self.video_analizer._get_config({'interval':[(0,1e10)], 'frames_per_sample':1000})
        x, _ = self.video_analizer.process_video(files=[path_to_video], config=config, rich_features=self.rich)
        if len(x)==0:      raise Exception(f'0 samples have been extracted from {path_to_video}')
        
        y = self.predict(x)
        y = [max(0,v) for v in y]
        y = sum(y)/len(y)
        label = (y>0.5 and 'real' or 'fake')

        if landmark_video:
            save_video_landmarks(path_to_video, label, self.video_analizer)

        return return_label and label or y

    def fit(self, x):
        self.clf.fit(x)
        return self

    def predict(self, x):
        return self.clf.predict(x)


class BoostedOneClassRbf(OneClassRbf):
    def __init__(self, video_analizer, rich, person):
        self.video_analizer = video_analizer
        self.clf1 = svm.OneClassSVM(nu=0.2, kernel="rbf", gamma='auto')
        self.clf2 = svm.OneClassSVM(gamma='auto')
        f_path = pathlib.Path(__file__).parent.absolute().joinpath(f'pretrainedSVC_{rich}_{person}.joblib')
        self.clf3 = joblib.load(f_path)
        self.rich = rich

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
