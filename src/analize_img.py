from openface import OpenFaceAPI
import openface.parts as parts
import cv2 as cv

# This file show some code snippets on HOW TO USE the openface module

PATH_TO_OPENFACE_DIR = '/home/tom/Desktop/OpenFace'  # where i cloned OpenFace repo 
img_to_process = '../test_data/img/selfie.jpg'       # image to process...


PATH_TO_OPENFACE_DIR=None ### if no argument   ==> ask to install
# setup API to call OpenFace functions
openfaceAPI = OpenFaceAPI(PATH_TO_OPENFACE_DIR)


# process an image
results = openfaceAPI.process_images(files=[img_to_process])
results = results[img_to_process.split('/')[-1]]
parts_of_face = [
    parts.FACE_CHIN,         # face shape
    parts.EYES_RIGHT_SCLERA,  # right eye
    parts.EYES_LEFT_SCLERA  # right eye
]

# print features
print(results.to_array())                                           # every feature
print(results.get_features(parts_of_face).to_array())               # selecting features
print(results.get_face(0).get_features(parts_of_face).to_array())   # selecting features only of the first face



### load image landmarks
img = cv.imread(img_to_process)

"""
###### get & show 3D points
points = results.get_landmarks(parts_of_face, dimension='3D')
points += 200  

for i in range(points.shape[1]):
    x,y = int(points[0][i]), int(points[1][i])
    img[y-2:y+2,x-2:x+2] = [0,255,0]
"""
###### get & show 2D points
points = results.get_landmarks(parts_of_face, dimension='2D')

for i in range(points.shape[1]):
    x,y = int(points[0][i]), int(points[1][i])
    img[y-2:y+2,x-2:x+2] = [0,200,255]


# show image
cv.imshow('face', img)
cv.waitKey(0)