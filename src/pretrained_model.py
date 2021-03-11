import os, pathlib, joblib
from videoanalizer import VideoAnalizer
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

RICH_FEATURES = [0,1,2] # 0 -> 190, 1 -> 250, 2 -> only rich features (60)

REMOVE_PERSON = ["ElonMusk", "Obama"]

results = {}

for rich in RICH_FEATURES:
    for person in REMOVE_PERSON:
        vd = VideoAnalizer()

        reals = [x for x in os.listdir('../test_data/videos/real') if '__' not in x]
        fakes = [x for x in os.listdir('../test_data/videos/fake') if '__' not in x]
        fakes.remove(person)
        for i,r in enumerate(reals[:]):
            if(r in REMOVE_PERSON):
                reals[i] = f"{r}/train"
        R = []
        F=[]
        for d in reals:
            v, _ = vd.process_video(fdir='../test_data/videos/real/'+d, rich_features=rich, config={'frames_per_sample':600, 'overlap':2, 'only_success':True})
            R += v
        for d in fakes:
            v, _ = vd.process_video(fdir='../test_data/videos/fake/'+d, rich_features=rich, config={'frames_per_sample':600, 'overlap':2, 'only_success':True})
            F += v

        X = R+F
        Y = [1]*len(R)+[-1]*len(F)

        clf = SVC(kernel='rbf')
        clf.fit(X,Y)

        y_pred = clf.predict(X)
        results[rich] = confusion_matrix(Y, y_pred)

        joblib.dump(clf, f'videoanalizer/pretrainedSVC_{rich}_{person}.joblib')

print(results)
