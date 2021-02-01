import os, pathlib
from gdrive import GoogleDriveDownloader as gdd

os.chdir(pathlib.Path(__file__).parent.absolute())

DATA_DIR = '../test_data'
SAVE_CLF_PATH = '../output/fake-real.joblib'
GDRIVE_CODE = '1JZ0lvEQmyXAxjJSZLFRx6WfF1w17NX6c'


######## Create Directory & Download Videos

os.system(f'mkdir {DATA_DIR}')
if (len(os.listdir(DATA_DIR))==0):
    # if folder is empty, get videos from gdrive
    gdd.download_file_from_google_drive(file_id=GDRIVE_CODE,    # my GDrive
                                        dest_path=DATA_DIR+'/dataset.zip',
                                        unzip=True)
    os.system(f'rm {DATA_DIR}/dataset.zip')
