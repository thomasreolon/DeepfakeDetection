from videoanalizer.openface.output_extractor import VidDataExtractor



def extract_samples(dx:VidDataExtractor, config:dict):
    if not all(setting in config for setting in ()):
        raise Exception("configs must contain 'interval', 'frames_per_sample','overlap' and 'only_success")




"""
TODO:

- openface output ---> cache + wrong path
- implement extract samples & covariance
- function to store processed videos features
"""