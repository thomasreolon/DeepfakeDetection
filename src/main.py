from openface import OpenFaceAPI
import openface.parts as parts
import cv2 as cv
import numpy as np

"""
this file show some code snippets on HOW TO USE the openface module
"""

PATH_TO_OPENFACE_DIR = '/home/tom/Desktop/OpenFace'  # where i cloned OpenFace repo 
img_to_process = '../test_data/img/selfie.jpg'       # image to process...


# setup API to call OpenFace functions
openfaceAPI = OpenFaceAPI(PATH_TO_OPENFACE_DIR)


# process an image
results = openfaceAPI.get_faceLand(img_to_process)
face_parts = results.get_features([parts.FACE_CHIN])



# show what the model extracted 
img = cv.imread(img_to_process)
for v in (0,1):
    points = np.array(face_parts.csv.iloc[v].array).copy().reshape(2,-1)
    for x,y in zip(points[0], points[1]):
        x,y = int(x), int(y)
        img[y-2:y+2,x-2:x+2] = [0,255,0]

cv.imshow('face', img)
cv.waitKey(0)