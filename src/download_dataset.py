import os, pathlib
from gdrive import GoogleDriveDownloader as gdd
from videoanalizer import VideoAnalizer

os.chdir(pathlib.Path(__file__).parent.absolute())

#GDRIVE_CODE = '1La_4SVYRNT8ePgLzZGwlxuY1V9x989DC'  # old dataset
GDRIVE_CODE = '1djFwp9vLkmOtd65ylXSYYn7C1ju08lJ7'
DATA_DIR = '../test_data'

######## Create Directory & Download Videos

os.system(f'mkdir {DATA_DIR}')
if (len(os.listdir(DATA_DIR))==0):
    # if folder is empty, get videos from gdrive
    print("Default dataset size: 6GB")
    gdd.download_file_from_google_drive(file_id=GDRIVE_CODE,    # my GDrive
                                        dest_path=DATA_DIR+'/dataset.zip',
                                        unzip=True,
                                        showsize=True)
    os.system(f'rm {DATA_DIR}/dataset.zip')



######## This will fail and ask you to install OpenFace

# press y to install OpenFace
VideoAnalizer()
