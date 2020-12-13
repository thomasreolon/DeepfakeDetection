###########################
# 
#  this class can compare the similarity between two faces
#
#  typical usage: if there are multiple people in a video, 
#  but we are interested only in one of them,
#  we can compare each face found in the video with a picture of our subject
#
##########################

import numpy as np


class FaceComparator():
    """This class is used to compare the similarity between two faces"""

    def __init__(self, data_extractor, partial_path):
        self.partial_path = partial_path
        self.data_extractor = data_extractor
        self.cache = {}

    def get_difference_score(self, faceid, path_to_face):
        """compare how different are the two faces, lower values <--> similar faces"""
        df = self.data_extractor.csv

        face1 = df.loc[(df['confidence']>0.9) & (df['face_id']==faceid)]

        if (len(df)==0):
            return 1e10

        face2 = self.data_extractor.openfaceAPI.process_images(files=[path_to_face])
        face2 = face2[path_to_face.split('/')[-1]]

        s1 = self.score(face1)
        s2 = self.score(face2)
        diff = s1-s2
        return diff.dot(diff)

    def score(self, df):
        df = df.sample(n=1)
        d_eye1   = self.distance(36,45,df)
        d_eye2   = self.distance(39,42,df)
        d_nose1  = self.distance(27,28,df)
        d_nose2  = self.distance(28,30,df)
        d_chin1  = self.distance(0,8,df)
        d_chin2  = self.distance(3,13,df)
        d_mouth1 = self.distance(57,8,df)
        d_mouth2 = self.distance(54,12,df)

        return np.array([d_eye1,d_eye2,d_nose1,d_nose2,d_chin1,d_chin2,d_mouth1,d_mouth2])

    def distance(self, a, b, df):
        cols = [f'X_{a}', f'Y_{a}', f'Z_{a}', f'X_{b}', f'Y_{b}', f'Z_{b}']
        arr  = df[cols].values.tolist()
        p1, p2 = np.array(arr[:3]), np.array(arr[-3:])
        diff = p1 - p2
        return diff.dot(diff)


        

