###########################
# 
#  helper classes that defines how to get the pieces of the csv
#
#  csv format: https://github.com/TadasBaltrusaitis/OpenFace/wiki/Output-Format
#
##########################


"""Shortcuts to selects parts of the face"""

# face
FACE = (0, 67, '')

FACE_CHIN          = (0, 16, '')
FACE_RIGHT_EYEBROW = (17,21, '')
FACE_LEFT_EYEBROW  = (22,26, '')
FACE_NOSE          = (27,35, '')
FACE_RIGHT_EYE     = (36,41, '')
FACE_LEFT_EYE      = (42,47, '')
FACE_OUTER_MOUTH    = (48,59, '')
FACE_INNER_MOUTH    = (60,67, '')



# eye_detail
EYES = (0, 55, 'eye_lmk_')

EYES_RIGHT_IRIS   = (0,7, 'eye_lmk_')
EYES_RIGHT_SCLERA = (8,19, 'eye_lmk_')
EYES_RIGHT_PUPIL  = (20,27, 'eye_lmk_')
EYES_LEFT_IRIS    = (28,35, 'eye_lmk_')
EYES_LEFT_SCLERA  = (36,47, 'eye_lmk_')
EYES_LEFT_PUPIL   = (48,55, 'eye_lmk_')


# gaze
GAZE = ['gaze_0_x','gaze_0_y','gaze_0_z','gaze_1_x','gaze_1_y','gaze_1_z','gaze_angle_x','gaze_angle_y']
CONFIDENCE = ['confidence']


# pose
POSE_COORD    = ['pose_Tx','pose_Ty','pose_Tz']
POSE_ROTATION = ['pose_Rx','pose_Ry','pose_Rz']




# p
P_GENERAL = ['p_scale','p_rx','p_ry','p_rz','p_tx','p_ty']
P_BOOO = [f'p_{i}' for i in range(34)]


# AU
AU_INTENSITIES  = ['AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU07_r', 'AU09_r', 'AU10_r', 'AU12_r', 'AU14_r', 'AU15_r', 'AU17_r', 'AU20_r', 'AU23_r', 'AU25_r', 'AU26_r', 'AU45_r']
AU_PRESENCE     = ['AU01_c', 'AU02_c', 'AU04_c', 'AU05_c', 'AU06_c', 'AU07_c', 'AU09_c', 'AU10_c', 'AU12_c', 'AU14_c', 'AU15_c', 'AU17_c', 'AU20_c', 'AU23_c', 'AU25_c', 'AU26_c', 'AU28_c', 'AU45_c']
AU              = AU_INTENSITIES + AU_PRESENCE


