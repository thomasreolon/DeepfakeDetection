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

import typing
import errno
import os
from . import errors as M
from . import output_extractor


# TODO:  fix       VIDIOC_REQBUFS: Inappropriate ioctl for device
# opencv do not support .mp4 ????

"""
VIDIOC_REQBUFS: Inappropriate ioctl for device  ===> rebuild OPENCV


I have solved this issue on Ubuntu 16.04.3.

    sudo apt-get install ffmpeg
    sudo apt-get install libavcodec-dev libavformat-dev libavdevice-dev

    Rebuild OpenCV 3.3.0 with the following commands:
        cd build
        cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local  -D WITH_FFMPEG=ON -D WITH_TBB=ON -D WITH_GTK=ON -D WITH_V4L=ON -D WITH_OPENGL=ON -D WITH_CUBLAS=ON -DWITH_QT=OFF -DCUDA_NVCC_FLAGS="-D_FORCE_INLINES" ..
        make -j7
        sudo make install


"""

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
        self.out_dir = out_dir
        try:
            os.mkdir(out_dir)
        except OSError as err:
            if err.errno != errno.EEXIST: raise err

    def process_images(self, files:list=None, fdir:str=None, bboxdir=None, out_dir:str=None):
        """
        Function used to compute landmarks of images and saves the results in out_dir
        input:  
            files(list of .jpg .png .bmp .jpeg files)
            out_dir(path where output is stored)

        output:
            dict( filename-->class_for_extracting_data_from_that_filename )

        files generated:
            fname_aligned, fname.csv, fname.hog, fname.jpg, fname_of_details.txt
        """
        if files is None and fdir is None:
            raise Exception(M.EXE_PROCESS_IMG)
        
        # get path where outputs are saved
        if not out_dir:
            out_dir = self.out_dir
        else:
            out_dir = self._get_abs_path(out_dir)

        # get source arguments
        if fdir:
            fdir = self._get_abs_path(fdir)
            formats = ['.jpg', '.png', '.bmp', 'jpeg']
            paths = [f for f in os.listdir(fdir) if f[-4:] in formats]
            src = f'-fdir {fdir}'
        else:
            files = [self._get_abs_path(f) for f in files]
            paths = [f.split('/')[-1] for f in files]
            src = ' '.join([f'-f {f}' for f in files])

        # bbox
        bbox = (bboxdir and f'-bboxdir {bboxdir}') or ''

        # create OpenFace cmd
        cmd = f"{self.exe_path}/FaceLandmarkImg {src} -out_dir {out_dir} {bbox}"

        # execute it
        os.system(cmd)

        res = {}
        for p in paths:
            p_path = os.path.join(out_dir, p.split('.')[0])
            res[p] = output_extractor.DataExtractor(p_path, self)
            # res['lena.jpg'] = DataExtractor('/tmp/openfacesaves/lena', self)

        return res

    def process_video(self, files:list=None, fdir:str=None, vtype='multi', add_param='', out_dir:str=None):
        """
        Function used to compute landmarks of videos and saves the results in out_dir
        
        input:

            vtype= 'multi' | 'single'    (number of people in the video)
        """
        if files is None and fdir is None:
            raise Exception(M.EXE_PROCESS_VID)
        assert vtype in ('single', 'multi')

        if vtype=='multi':
            exe = 'FaceLandmarkVidMulti'
        else:
            exe = 'FaceLandmarkVid'

        formats = ('.avi')
        if fdir and vtype=='single':
            fdir = self._get_abs_path(fdir)
            files = [f'-f {f}' for f in os.listdir(fdir) if f[-4:] in formats]
            paths = [f for f in os.listdir(fdir) if f[-4:] in formats]
            src = f"-inroot {fdir} {' '.join(files)}"
        elif fdir:
            fdir = self._get_abs_path(fdir)
            formats = ['.jpg', '.png', '.bmp', 'jpeg']
            paths = [f for f in os.listdir(fdir) if f[-4:] in formats]
            src = f'-fdir {fdir}'
        else:
            files = [self._get_abs_path(f) for f in files]
            paths = [f.split('/')[-1] for f in files]
            src = ' '.join([f'-f {f}' for f in files if f[-4:] in formats])

        if len(src)<=6:
            raise Exception(M.EXE_PROCESS_VID_FILES)

        cmd = f"{self.exe_path}/{exe} {src} -out_dir {out_dir} {add_param}"

        # execute it
        os.system(cmd)



    def get_faceLand(self, src:str, out_dir:typing.Union[str,None]=None):
        """
        Take a image conpute landmarks and saves the results in out_dir
        input:  src(path to a image file .jpg .png .bmp .jpeg)
                out_dir(path where output is stored)

        files generated:
            fname_aligned, fname.csv, fname.hog, fname.jpg, fname_of_details.txt
        """
        # get path where outputs are saved
        if not out_dir:
            out_dir = self.out_dir
        else:
            out_dir = self._get_abs_path(out_dir)

        # create OpenFace cmd
        src = self._get_abs_path(src)
        cmd = f"{self.exe_path}/FaceLandmarkImg -f {src} -out_dir {out_dir}"

        # execute cmd
        os.system(cmd)

        # where the results are stored
        fname = src.split('/')[-1].split('.')[0]
        partial_path = os.path.join(out_dir, fname)
        return output_extractor.DataExtractor(partial_path, self)




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









