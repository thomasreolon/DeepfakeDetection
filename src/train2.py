import os
from joblib import dump
from sklearn import svm
from sklearn.metrics import confusion_matrix
from videoanalizer import VideoAnalizer
from gdrive import GoogleDriveDownloader as gdd

"""

this code takes 2 folders,
train a classifier to distinguish the 2 folders

default classifier is OneClassSVM ==> merge the folders and tries to guess outliers

"""

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
clf = vd.train_classifier(f'{DATA_DIR}/real', f'{DATA_DIR}/fake', show_trainig_performance=True)
clf.predict_video('../test_data/vid/obama2.mp4', prints=True)

vd.save_classifier(clf)