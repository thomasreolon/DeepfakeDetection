from sklearn import svm

class OneClassRbf():
    def __init__(self, video_analizer, rich):
        self.video_analizer = video_analizer
        self.clf = svm.OneClassSVM(kernel='rbf', gamma='scale')
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
        x, _ = self.video_analizer.process_video(files=[path_to_video], config=config, rich=self.rich)
        y = self.predict(x)[0]

        if (return_label): y = (y==1 and 'real' or 'fake')
        return y

    def fit(self, x):
        self.clf.fit(x)
        return self
        
    def predict(self, x):
        return self.clf.predict(x)
