###########################
# 
#  interface for using OpenFace from Python
#  this class exposes methods that process images/videos
#
#  EXAMPLE in linux env:
#    $ echo "Installing OpenFace"
#    $ cd /home/user/Desktop && git clone https://github.com/TadasBaltrusaitis/OpenFace.git
#    $ cd OpenFace
#    $ sudo bash download_models.sh
#    $ sudo bash install.sh
#    $ 
#    $ echo "Using OpenFace"
#    $ python3
#    >>> import openface    # the module implemented in this repo
#    >>> api = openface.OpanFaceAPI('/home/user/Desktop/OpenFace')
#    >>> api.get_faceLand('image.jpg')
#
##########################

import os
from . import errormessages as M
import cv2 as cv
import torch

class OpenFaceAPI():
    """
This class serves as an interface between OpenFace executables and Python.
given the name of the folder of openface (abs path), this class will simplify the calling of the functions.
    """

    def __init__(self, openface_path:str, out_dir:str='/tmp/openfacesaves'):
        # Get path where openface executables should be
        exe_path = os.path.join(openface_path, 'build/bin') # path where openface executables are
        self.exe_path = exe_path

        # Check that the OpenFace installation folder is found at openface_path
        self._check_init_files(exe_path, openface_path)

        # Create Folder to save OpanFace results
        out_dir = self._get_abs_path(out_dir)
        self.out_dir = out_dir
        os.system(f'mkdir -p {out_dir}')

    def get_faceLand(self, src:str, out_dir:str|None=None):
        """
        Take a image conpute landmarks and saves the results in out_dir
        input:  src(path to a image file .jpg .png .bmp .jpeg)
                out_dir(path where output is stored)
        """
        # get path where outputs are saved
        if not out_dir:
            out_dir = self.out_dir
        else:
            out_dir = self._get_abs_path(out_dir)

        # create OpenFace cmd
        src = self._get_abs_path(src)
        CMD = f"{self.exe_path}/FaceLandmarkImg -f {src} -out_dir {out_dir}"

        # execute cmd
        try:
            os.system(CMD)
            print(f'--> computed {src}, results in {out_dir}')
        except Exception as e:
            print(M.EXE_LANDMARK_IMG.format(CMD, e))




    def _check_init_files(self, exe_path, openface_path):
        """Check that the path given in input is valid"""

        # Check that openface_path is correct
        # in theory Openface/exe/FaceLandmarkImg should contain FaceLandmarkImg.cpp
        check_path = os.path.join(openface_path, 'exe/FaceLandmarkImg/FaceLandmarkImg.cpp')
        if not os.path.isfile(check_path):
            raise Exception(M.INIT_INVALID_PATH.format(openface_path))
        
        # Check that executables are present
        files = os.listdir(exe_path) # files that we have
        needed = ['FaceLandmarkImg', 'FaceLandmarkVid', 'FaceLandmarkVidMulti', 'FeatureExtraction'] # files that we need
        if not all([f in files for f in needed]):
            raise Exception(M.INIT_EXE_NOT_FOUND.format(exe_path))

        # Check that the models have been downloaded
        check_path = os.path.join(openface_path,'lib/local/LandmarkDetector/model/patch_experts/cen_patches_0.50_of.dat')
        if not os.path.isfile(check_path):
            raise Exception(M.INIT_MODELS_NOT_FOUND)

    def _get_abs_path(self, path:str)->str:
        """Check that path exists & make it absolute"""

        # check if path exists
        if not os.path.exists(path):
            raise Exception(M.PATH_NOT_FOUND.format(path))

        # if it is relative, make it absolute
        if not os.path.isabs(path):
            path = os.path.join(os.getcwd(), path)
        
        return path









