import argparse
from joblib import load
from videoanalizer import VideoAnalizer

SAVE_CLF_PATH = '../output/fake-real.joblib'


####### Handle User's input

#extracting arguments from command line
#--vn video name to process
#--part the part to display (like FACE, EYES ...)
#--objtocompare the object to track to --part object)
parser = argparse.ArgumentParser(description='A program to process videos')
parser.add_argument('--vn', help='the video\'s name, it has to be inside <project>/test_data/vid (DEFAULT = obama2.mp4)', default="obama2.mp4")
#parser.add_argument('--part', help='the video\'s parts to process, can be: FACE (FACE_CHIN, FACE_RIGHT_EYEBROW ...), EYES (EYES_RIGHT_IRIS, ...), see parts.py for more (DEFAULT = FACE)', default="FACE")
#parser.add_argument('--objtotrack', help='the object to track (compare) with --part\'s object \n (DEFAULT = andrew.jpeg andrew\'s face)', default="andrew.jpeg")
args = parser.parse_args()

video_name = '../test_data/vid/' +args.vn
#part_to_process = args.part
#object_to_track = args.objtotrack


####### Analyze Input Video

vd = VideoAnalizer()
x = vd.process_video(files=[video_name])



####### Load Classifier and Predict

clf = load(SAVE_CLF_PATH)
y = clf.predict(x)

print(f'predicted:{y[0]}')
