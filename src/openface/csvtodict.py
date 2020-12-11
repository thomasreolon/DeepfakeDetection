###########################
# 
#  OpenFace produces a .csv containing the informations about a face.
#  this file reads these csvs and return a more readable data structure
#
##########################





class ImageLandmarks:
    confidence:float = 0.
    eyes:list = None
    # p, z, x, y, au, gaze


class CSVReader():
    def read_image_landmark(self, path):
        
        with open(path, 'r') as fin:
            content = fin.read()


