import cv2 as cv
import openface
import openface.parts as parts

PATH_TO_OPENFACE_DIR = '/home/tom/Desktop/OpenFace'  # where i cloned OpenFace repo 
vid_to_process = '../test_data/vid/manypeople.mp4'       # image to process...
andrew_face = '../test_data/img/andrew.jpeg'


# setup API to call OpenFace functions
openfaceAPI = openface.OpenFaceAPI()

results = openfaceAPI.process_video(files=[vid_to_process], vtype='single')
filename = vid_to_process.split('/')[-1]
results = results[filename]


#  show output
cap = cv.VideoCapture(vid_to_process)

print(results.get_face(0).get_features([parts.FACE]).to_array())   # selecting features only of the first face

results = results.get_face(andrew_face)


############# show labels
fr=0
while True:
    isTrue, frame = cap.read()

    ###### adding labels
    try:
        points = results.get_landmarks(frame=fr, face_parts=[parts.FACE])

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
