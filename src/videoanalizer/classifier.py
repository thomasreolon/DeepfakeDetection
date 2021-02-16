from sklearn import svm
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
class Classifier():
    def __init__(self, clf_type, labels_map=None, video_analizer=None):
        self.labels_map = labels_map
        self.video_analizer = video_analizer

        if (clf_type=='OneClassSVM'):
            self.clf = svm.OneClassSVM(max_iter=10000)
        elif(clf_type=='LinearSVM'):
            self.clf = svm.LinearSVC()
        elif(clf_type=='SVC'):
            self.clf = svm.SVC(kernel='rbf')
        elif(clf_type=='KDTree'):
            self.clf = KNeighborsClassifier(n_neighbors=3)
        elif(clf_type=='RandomForestRegressor'):
            self.clf = RandomForestRegressor(n_estimators=30, random_state=0)
        elif(clf_type=='DecisionTreeClassifier'):
            self.clf = DecisionTreeClassifier()
        else:
            raise Exception(f'{clf_type} clf type non valid, must be in [OneClassSVM, LinearSVM, SVC, KDTree, RandomForestRegressor, DecisionTreeClassifier]')


    def set_labels_map(self, labels_map):
        """save association between classifier output and real class behind (eg. 1->Real, -1->Fake)"""
        self.labels_map = labels_map

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
        x, _ = self.video_analizer.process_video(files=[path_to_video], config=config)
        y = self.predict(x)[0]

        if (return_label): y = self.labels_map[y]
        return y

    def fit(self, x,y):
        self.clf.fit(x,y)
        return self
        
    def predict(self, x):
        return self.clf.predict(x)



def train_specific_person_classifier(person_samples, non_person_samples, video_analizer, person_name='Real', clf_type='OneClassSvm', show_trainig_performance=False):
    """
    clf_type: OneClassSvm, LinearSVM, SVC, KDTree, RandomForestRegressor, DecisionTreeClassifier
    """
    
    # Build Train Input
    x = person_samples + non_person_samples
    y_true = [1 for _ in person_samples] + [-1 for _ in non_person_samples]

    # Do The Search for the Best Classifier
    lm = {1:person_name, -1:'Non '+person_name}
    clf = Classifier(clf_type, labels_map=lm, video_analizer=video_analizer)
    if (show_trainig_performance): print('training....')
    clf = clf.fit(x, y_true)

    # Print Results on training test itself
    if (show_trainig_performance):
        y_pred = clf.predict(x)
        report = metrics.classification_report(y_true, y_pred, zero_division=0)
        print('Showing if the classifier learned the training inputs')
        print(report)

    return clf

