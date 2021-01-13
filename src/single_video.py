import cv2 as cv, openface, openface.parts as parts, argparse, math, openface.library as library

#extracting arguments from command line
#--vn video name to process
#--part the part to display (like FACE, EYES ...)
#--objtocompare the object to track to --part object)
parser = argparse.ArgumentParser(description='A program to process videos')
parser.add_argument('--vn', help='the video\'s name, it has to be inside <project>/test_data/vid (DEFAULT = manypeople.mp4)', default="manypeople.mp4")
parser.add_argument('--part', help='the video\'s parts to process, can be: FACE (FACE_CHIN, FACE_RIGHT_EYEBROW ...), EYES (EYES_RIGHT_IRIS, ...), see parts.py for more (DEFAULT = FACE)', default="FACE")
# parser.add_argument('--objtotrack', help='the object to track (compare) with --part\'s object \n (DEFAULT = andrew.jpeg andrew\'s face)', default="andrew.jpeg")
args = parser.parse_args()

video_name = args.vn
part_to_process = args.part
# object_to_track = args.objtotrack

PATH_TO_OPENFACE_DIR = '/home/tom/Desktop/OpenFace'  # where i cloned OpenFace repo
vid_to_process = f'../test_data/vid/{video_name}'   # video to process...

# process video
api = openface.OpenFaceAPI()

results = api.process_video([vid_to_process], vtype='single')

for video_name, extractor in results.items():
    features = extractor.get_raw_features(AU_r = parts.AU_paper_r,
                                          AU_c = parts.AU_paper_c,
                                          pose = parts.POSE_ROTATION_X_Z,
                                          mouth_h = parts.MOUTH_H,
                                          mouth_v = parts.MOUTH_V)

features_matrix = []

for f0 in features:
    f0 = features[f0]
    row = []
    for f1 in features:
        f1 = features[f1]
        row.append(library.pearson_correlation(f0, f1))
    features_matrix.append(row)

print(features_matrix)
