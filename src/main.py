from openface import OpenFaceAPI

PATH_TO_OPENFACE_DIR = '/home/tom/Desktop/OpenFace'



# setup API to call OpenFace functions
openfaceAPI = OpenFaceAPI(PATH_TO_OPENFACE_DIR)


# process an image
results = openfaceAPI.get_faceLand('../test_data/img/selfie.jpg')

print(results.csv.head())

