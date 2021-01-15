from sklearn.svm import OneClassSVM
from sklearn.model_selection import GridSearchCV
from sklearn import metrics

class SVMClassifier(GridSearchCV):
    def __init__(self, *args, **kw):
        self.labels_map = ('labels_map' in kw and kw['labels_map'])or None
        self.video_analizer = ('video_analizer' in kw and kw['video_analizer'])or None
        del kw['labels_map']
        del kw['video_analizer']
        super().__init__(*args, **kw)

    def set_labels_map(self, labels_map):
        """save association between classifier output and real class behind (eg. 1->Real, -1->Fake)"""
        self.labels_map = labels_map

    def set_video_analizer(self, video_analizer):
        """save association between classifier output and real class behind (eg. 1->Real, -1->Fake)"""
        self.video_analizer = video_analizer

    def predict_video(self, path_to_video, prints=False):
        """
        input:
            - video path
        oputput:
            - integer that maps 'labels_map' into person is real or not
        """
        config = self.video_analizer._get_config({'interval':[(0,1e10)], 'frames_per_sample':-1})
        x = self.video_analizer.process_video(files=[path_to_video], config=config)
        y = self.predict(x)[0]

        if (prints): print('result ---> ',self.labels_map[y])
        return y


def train_specific_person_classifier(person_samples, non_person_samples, video_analizer, person_name='Real', show_trainig_performance=False)->SVMClassifier:
    # Build Train Input
    x = person_samples + non_person_samples
    y_true = [1 for _ in person_samples] + [-1 for _ in non_person_samples]

    # Params to Try
    params={
        'gamma':['auto','scale'],
    }

    # Do The Search for the Best Classifier
    lm = {1:person_name, -1:'Non '+person_name}
    svm = OneClassSVM(max_iter=10000, kernel='rbf')
    clf = SVMClassifier(svm, params, cv=3, scoring="accuracy", labels_map=lm, video_analizer=video_analizer)
    if (show_trainig_performance): print('training....')
    clf = clf.fit(x, y_true)

    # Print Results on training test itself
    if (show_trainig_performance):
        y_pred = clf.predict(x)
        report = metrics.classification_report(y_true, y_pred, zero_division=0)
        print('Showing if the classifier learned the training inputs')
        print(report)

    return clf

