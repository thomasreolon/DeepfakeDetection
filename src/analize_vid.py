import cv2 as cv, openface, openface.parts as parts, argparse

#extracting arguments from command line
#--vn video name to process
#--part the part to display (like FACE, EYES ...)
#--objtocompare the object to track to --part object)
parser = argparse.ArgumentParser(description='A program to process videos')
parser.add_argument('--vn', help='the video\'s name, it has to be inside <project>/test_data/vid (DEFAULT = manypeople.mp4)', default="manypeople.mp4")
parser.add_argument('--part', help='the video\'s parts to process, can be: FACE (FACE_CHIN, FACE_RIGHT_EYEBROW ...), EYES (EYES_RIGHT_IRIS, ...), see parts.py for more (DEFAULT = FACE)', default="FACE")
parser.add_argument('--objtotrack', help='the object to track (compare) with --part\'s object \n (DEFAULT = andrew.jpeg andrew\'s face)', default="andrew.jpeg")
args = parser.parse_args()

video_name = args.vn
part_to_process = args.part
object_to_track = args.objtotrack

PATH_TO_OPENFACE_DIR = '/home/tom/Desktop/OpenFace'  # where i cloned OpenFace repo
vid_to_process = f'../test_data/vid/{video_name}'   # video to process...
andrew_face = f'../test_data/img/{object_to_track}' # object to track in the video


# setup API to call OpenFace functions
openfaceAPI = openface.OpenFaceAPI()

results = openfaceAPI.process_video(files=[vid_to_process], vtype='single')
filename = vid_to_process.split('/')[-1]
results = results[filename]


#  show output
cap = cv.VideoCapture(vid_to_process)

print(results.get_face(0).get_features([getattr(parts, part_to_process)]).to_array())   # selecting features only of the first face

results = results.get_face(andrew_face)


############# show labels
fr=0
while True:
    isTrue, frame = cap.read()
    if(frame is None):
        break

    ###### adding labels
    try:
        points = results.get_landmarks(frame=fr, face_parts=[getattr(parts, part_to_process)])

        for i in range(points.shape[1]):
            x,y = int(points[0][i]), int(points[1][i])
            frame[y-2:y+2,x-2:x+2] = [0,200,255]
    except IndexError:
        print(f'IndexError (no records for that frame), failed frame:{fr}')
    except ValueError:
        print(f'ValueError (no records for that frame), failed frame:{fr}')


    cv.imshow(filename, frame)
    fr+=1

    if cv.waitKey(20) & 0xFF==ord('d'):
        break

cap.release()
cv.destroyAllWindows()
