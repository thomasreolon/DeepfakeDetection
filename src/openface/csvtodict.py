###########################
# 
#  OpenFace produces a .csv containing the informations about a face.
#  this file reads these csvs and return a more readable data structure
#
##########################





class ImageLandmarks:           ###### needs tesing to decide what is what
    confidence:float = 0.
    eyes:list = None
    # p, z, x, y, au, gaze

def select(data, xpar, ypar):
    X, Y = [], []
    for p,v in data:
        if xpar in p:
            X.append(v)
    for p,v in data:
        if ypar in p:
            Y.append(v)
    return X, Y


######  read csv: https://github.com/TadasBaltrusaitis/OpenFace/wiki/Output-Format


class DataExtractor():
    """
    This class reads the data that OpenFace writes on files and present
    it in a more readable way
    """

    def __init__(self, partial_path):
        """set where infos are stored
        partial_path = f'{out_dir}/{filename}'"""
        self.partial_path = partial_path

    def _load_csv(self):
        csv_path = self.partial_path + '.csv'




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

