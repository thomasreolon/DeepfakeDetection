import os
import pandas
import cv2
from .openface import output_extractor
from .openface.parts import FACE

def save_video_landmarks(video_path, result, video_analizer):

    # load files
    f_name = video_path.split('/')[-1].split('.')[0]
    csv = os.path.join(video_analizer.api.out_dir, f_name)
    df = output_extractor.VidDataExtractor(csv, video_analizer.api)
    out_file = os.path.join(video_analizer.api.out_dir, '../',f_name + '.avi')
    color = result=='real' and (0,255,0) or (0,0,255)

    # load video with opencv
    vid = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = vid.get(cv2.CAP_PROP_FPS)

    # get landmarks for each frame
    ret, out, i = True, None, 0
    while(vid.isOpened() and ret):
        # get frame
        ret, frame = vid.read()
        if ret:
            h, w = len(frame[0]),len(frame)
            if out is None:
                out = cv2.VideoWriter(out_file,fourcc, fps, (h,w))

            # get landmarks
            points = df.get_landmarks(face_parts=FACE, frame=i)
            i+=1
            if len(points)>0 and len(points[0])>0:
                min_x = int(min(points[0]))
                min_y = int(min(points[1]))
                max_x = int(max(points[0]))
                max_y = int(max(points[1]))


                # put landmarks in the frame
                tmp = frame[min_y+2:max_y-2, min_x+2:max_x-2].copy()
                frame[min_y:max_y, min_x:max_x] = color
                frame[min_y+2:max_y-2, min_x+2:max_x-2] = tmp

                # add label
                font                   = cv2.FONT_HERSHEY_SIMPLEX
                bottomLeftCornerOfText = (min_x, min_y-2)
                fontScale              = 1

                cv2.putText(frame, result, 
                    bottomLeftCornerOfText, 
                    font, 
                    fontScale,
                    color,
                    )

            # save on file
            out.write(frame)

    # save video
    vid.release()
    out.release()
    print(f'video saved in {out_file}')
