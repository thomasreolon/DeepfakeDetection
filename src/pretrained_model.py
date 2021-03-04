import os, pathlib, joblib
from videoanalizer import VideoAnalizer
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

vd = VideoAnalizer()

reals = [x for x in os.listdir('../test_data/videos/real') if '__' not in x]
fakes = [x for x in os.listdir('../test_data/videos/fake') if '__' not in x]
R = []
F=[]
for d in reals:
    v, _ = vd.process_video(fdir='../test_data/videos/real/'+d, config={'frames_per_second':600, 'overlap':2, 'only_success':True})
    R += v
for d in fakes:
    v, _ = vd.process_video(fdir='../test_data/videos/fake/'+d, config={'frames_per_second':600, 'overlap':2, 'only_success':True})
    F += v

X = R+F
Y = [1]*len(R)+[-1]*len(F)

clf = SVC(kernel='rbf')
clf.fit(X,Y)

y_pred = clf.predict(X)
print(confusion_matrix(Y, y_pred))

joblib.dump(clf, 'videoanalizer/pretrainedSVC.joblib')