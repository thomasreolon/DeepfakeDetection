import os
import cv2 as cv
import openface
import openface.parts as parts

PATH_TO_OPENFACE_DIR = '/home/tom/Desktop/OpenFace'  # where i cloned OpenFace repo 
vid_to_process = '../test_data/vid/obama2.mp4'       # image to process...



# setup API to call OpenFace functions
openfaceAPI = openface.OpenFaceAPI(PATH_TO_OPENFACE_DIR)

results = openfaceAPI.process_video(files=[vid_to_process], vtype='multi')
results = results[vid_to_process.split('/')[-1]]

