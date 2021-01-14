####################################################################################
#
#    extract the AU correlation features from a video
#    & train a classifier with them
#
#    --> note: in the paper a one-class SVM is trained
#
####################################################################################
import os
from joblib import dump
from sklearn import svm
from sklearn.metrics import confusion_matrix
from videoanalizer import VideoAnalizer
from gdrive import GoogleDriveDownloader as gdd


DATA_DIR = '../test_data/fake_vs_real'
SAVE_CLF_PATH = '../output/fake-real.joblib'

######## Create Directory & Download Videos

os.system(f'mkdir -d {DATA_DIR}')
if (len(os.listdir(DATA_DIR))==0):
    # if folder is empty, get videos from gdrive
    gdd.download_file_from_google_drive(file_id='1i1H4cWGVoNtYNCGKs7fr4nXwRPfbFJJw',    # my GDrive
                                        dest_path=DATA_DIR+'/data.zip',
                                        unzip=True, overwrite=False)
    os.system(f'rm {DATA_DIR}/data.zip')


######## Analysis of the Videos

vd = VideoAnalizer()
X, Y, labels = [], [], {}
for i, s in enumerate(os.listdir(DATA_DIR)):
    tmp = vd.process_video(fdir=f'{DATA_DIR}/{s}')
    X += tmp
    Y += [i for _ in range(len(tmp))]
    labels[i] = s

clf = svm.SVC()
clf.fit(X, Y)


######## Save Classifier

dump(clf, SAVE_CLF_PATH)



######## Print Labels & (if clf can classify the training set)

print('Note: just checking the performance on the training set (hopefully should be about 100%)')
print('\n\nLabels:')
for i, s in labels.items():
    print(f'|  {i} ---> {s}')

Ypred = clf.predict(X)
print('\n\nConfusion Matrix:\n',confusion_matrix(Y, Ypred))
