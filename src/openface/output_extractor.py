###########################
# 
#  OpenFace produces a .csv containing the informations about a face.
#  this file reads these csvs and return a more readable data structure
#
#  csv format: https://github.com/TadasBaltrusaitis/OpenFace/wiki/Output-Format
#
##########################

import os
import typing
import pandas as pd
import numpy as np
from . import face_comparator

class DataExtractor():
    """
    This class reads the data that OpenFace writes on files and present
    it in a more readable way
    """

    def __init__(self, partial_path:str, api, csv=None, hog=None, is_multi=False):
        """Set results directory & load csv
        partial_path = f'{out_dir}/{filename}'"""
        self.partial_path     = partial_path
        self.csv:pd.DataFrame = csv
        self.hog              = hog
        self.openfaceAPI      = api
        self.face_comparator  = face_comparator.FaceComparator(self, partial_path)
        self.fcol             = (is_multi and self.fcol) or 'face'
        if not isinstance(csv, pd.DataFrame):
            self.csv = pd.read_csv(partial_path + '.csv')


    def get_hog(self):
        """Load histogram of gradients file (to be implemented)"""
        if not self.hog:
            #hog_path = self.partial_path + '.hog'
            self.hog = None #...... implement how to read .hog..
        return self.hog

    def get_landmarks(self, face_parts:list, faceid:int=None, frame:int=0, dimension='2D' ):
        """Returns a DataExtractor containing only the columns from face_parts 
        input: 
            face_parts:    a list of prats of the face, only parts.FACE_ and parts.EYES are usable
            dimension:     '2D'|'3D'
            
        difference from   get_features: the result is an array shaped(-1, n_dim)
        """
        if not isinstance(face_parts, list):
            face_parts = [face_parts]
        df = self.csv

        # select requested frames
        if 'frame' in df:
            df = df.loc[df['frame']==frame]

        # select face
        if faceid is not None:
            df = df.loc[df[self.fcol]==faceid]

        # get coords
        res, n_dim = [], ((dimension=='2D' and 2) or 3)
        for fp in face_parts:
            cols = self._cmp((fp[0],fp[1]),n_dim==2,fp[2])

            for _, row in df[cols].iterrows():
                tmp = np.array(row.array).copy().reshape(n_dim, -1)
                res.append(tmp)
        #res = np.concatenate(tmp, axis=1) #not working
        X = [None]*n_dim
        for i in range(n_dim):
            tmp = [a[i] for a in res]
            X[i] = np.concatenate(tmp)
        X = np.array(X)

        return X  # x[dimension][point_id]



    def get_features(self, face_parts:list, dimension='2D'):
        """Returns a DataExtractor containing only the columns from face_parts 
        input: 
            face_parts:    a list of prats of the face, defined in openface.parts
            dimension:     '2D'|'3D'
            
        face_parts con include strings of single columns:[face, confidence, AU01_r, ...]   
        """
        if not isinstance(face_parts, list):
            face_parts = [face_parts]

        # order face_parts, so that:
        # [parts.EYES, parts.FACE] will generate the same result of [parts.FACE, parts.EYES]
        def mysort(fp):
            if isinstance(fp, list):
                return fp[0].__hash__()
            else:
                return fp.__hash__()
        face_parts.sort(key=mysort)

        # cols is the list of columns
        cols = [self.fcol, 'frame', 'confidence'] # these columns cannot be removed 
        for fp in face_parts:
            if isinstance(fp, str):
                cols.append(fp)
            elif isinstance(fp, tuple):
                cols += self._cmp((fp[0],fp[1]),dimension=='2D',fp[2])
            elif isinstance(fp, list):
                cols += fp
        
        # remove duplicates from columns
        cols = list(set(cols))

        # select only the requested columns
        df = self.csv[cols]
        return DataExtractor(self.partial_path, self.openfaceAPI, df, self.hog)

            
    def to_array(self, col_to_remove=None):
        """
        input:
            col_to_remove: list of cols that we want to remove

        output:
            {
                0:{               # face 0
                    0:{features...},  # frame 0
                    1:{features...},  # frame 1
                    5:{features...},  # frame 5
                },
                1:{...},          # face 1
            }"""
        if col_to_remove is None:
            col_to_remove = [self.fcol, 'frame', 'confidence']
        
        df = self.csv
        faces = set(df[self.fcol].tolist())

        list_of_faces = {}
        for face in faces:
            list_of_frames = {}
            df_face = df.loc[df[self.fcol]==face]              # get df where face == f
            if 'frame' in df_face:                             # CASE: VIDEO
                frames = set(df_face['frame'].tolist())
                for frame in frames:
                    tmp = df_face.loc[df_face['frame']==frame]  # get df where frame == fr & face == f
                    tmp = tmp.drop(col_to_remove, axis=1)       # select features
                    features = np.array(tmp.values.tolist())    # (if more than one dimensions line... many times same person in the same frame)
                    list_of_frames[frame] = features
                list_of_faces[face] = list_of_frames
            else:                                               # CASE: image
                if ('frame' in col_to_remove):
                    del col_to_remove[col_to_remove.index('frame')]
                df_face = df_face.drop(col_to_remove, axis=1)
                features = np.array(df_face.values.tolist()[0])
                list_of_faces[face] = features
        
        return list_of_faces

    def get_confidence(self, minthreshold:float,  maxthreshold:float=1.):
        """Returns a DataExtractor filtered by confidence >= minthreshold"""
        df = self.csv
        cond = (df['confidence'] >= minthreshold) & (df['confidence'] <= maxthreshold)
        df = df.loc[cond]
        return DataExtractor(self.partial_path, self.openfaceAPI, df, self.hog)

    def get_face(self, selector:typing.Union[int,str]):
        """Returns a DataExtractor filtered by confidence >= minthreshold"""
        df = self.csv
        if isinstance(selector, int):
            # selector is an id of the csv
            df = df.loc[df[self.fcol] == selector]
        elif isinstance(selector, str):
            # selectr is a path to a face
            if not os.path.isabs(selector):
                selector = os.path.join(os.getcwd(), selector)

            faces = set(df[self.fcol].tolist())
            min_face, min_score = -1, 1e10
            for face in faces:
                score = self.face_comparator.get_difference_score(face, selector)
                if (score < min_score):
                    min_face, min_score = face, score

            if (min_face!=-1):
                df = df.loc[df[self.fcol]==min_face]
            else:
                raise Exception('no faces found?')

        return DataExtractor(self.partial_path, self.openfaceAPI, df, self.hog)

    def get_frame(self, frame:int):
        """Returns a DataExtractor filtered by csv[frame] == frame"""
        df = self.csv
        df = df.loc[df['frame'] == frame]
        return DataExtractor(self.partial_path, self.openfaceAPI, df, self.hog)




    def _cmp(self, interval:tuple, is2D:bool, prev:str=''):
        """function to select specific columns of the CSV"""
        res = []
        
        # if points in 2D or 3D
        dimensions = ['X_', 'Y_', 'Z_']
        if is2D:
            dimensions = ['x_', 'y_']

        # build query: columns needed
        for d in dimensions:
            res += [f'{prev}{d}{i}' for i in range(interval[0], interval[1]+1)]

        return res

