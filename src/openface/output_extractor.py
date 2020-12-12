###########################
# 
#  OpenFace produces a .csv containing the informations about a face.
#  this file reads these csvs and return a more readable data structure
#
#  csv format: https://github.com/TadasBaltrusaitis/OpenFace/wiki/Output-Format
#
##########################

import pandas as pd




class DataExtractor():
    """
    This class reads the data that OpenFace writes on files and present
    it in a more readable way
    """

    def __init__(self, partial_path:str, csv=None, hog=None):
        """Set results directory & load csv
        partial_path = f'{out_dir}/{filename}'"""
        self.partial_path     = partial_path
        self.csv:pd.DataFrame = csv
        self.hog              = hog
        if not csv:
            self.csv = pd.read_csv(partial_path + '.csv')


    def get_hog(self):
        """Load histogram of gradients file (to be implemented)"""
        if not self.hog:
            hog_path = self.partial_path + '.hog'
            self.hog = None #...... implement how to read .hog..
        return self.hog

    def get_features(self, face_parts:list, dimension='2D'):
        """Get the parts of the face requested as an array
        input: 
            face_parts: a list of prats of the face, defined in openface.parts
            dimension: '2D'|'3D' 
            
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
        face_parts = face_parts.sort(key=mysort)


        res = []
        for fp in face_parts:
            if isinstance(fp, str):
                res.append(fp)
            elif isinstance(fp, tuple):
                res += self._cmp((fp[0],fp[1]),dimension=='2D',fp[2])
            elif isinstance(fp, list):
                res += fp

        df = self.csv[res]
        return 

            

        


        







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

