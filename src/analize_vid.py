import os
import cv2 as cv
import openface
import openface.parts as parts

PATH_TO_OPENFACE_DIR = '/home/tom/Desktop/OpenFace'  # where i cloned OpenFace repo 
img_to_process = '../test_data/vid/obama2.mp4'       # image to process...



# setup API to call OpenFace functions
openfaceAPI = openface.OpenFaceAPI(PATH_TO_OPENFACE_DIR)


