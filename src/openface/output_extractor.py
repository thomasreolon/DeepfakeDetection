###########################
# 
#  OpenFace produces a .csv containing the informations about a face.
#  this file reads these csvs and return a more readable data structure
#
#  csv format: https://github.com/TadasBaltrusaitis/OpenFace/wiki/Output-Format
#
##########################

import pandas as pd
import numpy as np
import typing

# This class is not completed
# TODO:
#   - implement to_array()
#   - create function get_frames()
#   - 1. implement a method (also in another module) to compare faces
#   - 2. use the method above to select the most similar face

class DataExtractor():
    """
    This class reads the data that OpenFace writes on files and present
    it in a more readable way
    """

    def __init__(self, partial_path:str, api, csv=None, hog=None):
        """Set results directory & load csv
        partial_path = f'{out_dir}/{filename}'"""
        self.partial_path     = partial_path
        self.csv:pd.DataFrame = csv
        self.hog              = hog
        self.openfaceAPI      = api
        if not isinstance(csv, pd.DataFrame):
            self.csv = pd.read_csv(partial_path + '.csv')


    def get_hog(self):
        """Load histogram of gradients file (to be implemented)"""
        if not self.hog:
            hog_path = self.partial_path + '.hog'
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
            df = df.loc[df['face']==faceid]

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
        cols = ['face', 'frame', 'confidence'] # these columns cannot be removed 
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

            
    def to_array(self, col_to_remove=['face', 'frame', 'confidence']):
        """
        input:
            col_to_remove: set of cols that we want to remove

        
        output:
            [
                [               # face 0
                    [features...],  # frame 0
                    [features...],  # frame 1
                    [features...],  # frame 2
                ],     
                [...],          # face 1
            ]"""
        pass

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
            df = df.loc[df['face'] == selector]
        elif isinstance(selector, str):
            # selectr is a path to a face
            if not os.path.isabs(selector):
                selector = os.path.join(os.getcwd(), selector)

            ####### IMPLEMENT foreach face in csv... compare it to image


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

        




    def read_image_landmark(self, path):
        """read image.csv from path and generate a class containing its informations"""

        # read file
        with open(path, 'r') as fin:
            content = fin.read()

        # 1st line: names of parameters, 2nd: values
        lines = content.split('\n')
        params = lines[0].split(',')
        vals = lines[1].split(',')

        data = [(p,v) for p, v in zip(lines, params)]
        


        # prova.... estrazione features...
        eye_x = select(data, 'eye_lmk_x', 'eye_lmk_y')
        eye_X = select(data, 'eye_lmk_X', 'eye_lmk_Y')

