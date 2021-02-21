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

import errno
import os
from . import output_extractor
from . import errors as M
from .installer import install_openface


class OpenFaceAPI():
    """
    This class serves as an interface between OpenFace executables and Python.
    given the name of the folder of openface (abs path), this class will simplify the calling of the functions.
    """

    def __init__(self, openface_path:str=None, out_dir:str=None):
        if out_dir is None:
            out_dir = '/'.join(str(__file__).split('/')[:-1])+'/../../../output/openfacesaves'

        if openface_path is None:
            openface_path = '/'.join(str(__file__).split('/')[:-1])+'/OpenFace'
            try:
                self._check_init_files(openface_path)
            except: # pylint: disable=bare-except
                install_openface()


        # Get path where openface executables should be
        exe_path = os.path.join(openface_path, 'build/bin') # path where openface executables are
        self.exe_path = exe_path

        # Check that the OpenFace installation folder is found at openface_path
        self._check_init_files(openface_path)

        # Create Folder to save OpanFace results
        self.out_dir = self._get_abs_path(out_dir)
        try:
            os.mkdir(out_dir)
        except OSError as err:
            try:
                os.mkdir('/'.join(str(__file__).split('/')[:-1])+'/../../../output')
                os.mkdir(out_dir)
            except OSError as err:
                if err.errno != errno.EEXIST: raise err

    def process_images(self, files:list=None, fdir:str=None, bboxdir=None, out_dir:str=None):
        """
        Function used to compute landmarks of images and saves the results in out_dir
        input:
            files(list of .jpg .png .bmp .jpeg files)
            fdir(string that represents a folder containing the videos)
            bboxdir(borderbox for videos, see OpenFace API)
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
            res[p] = output_extractor.ImgDataExtractor(p_path, self)
            # res['lena.jpg'] = DataExtractor('/tmp/openfacesaves/lena', self)

        return res

    def process_video(self, files:list=None, fdir:str=None, vtype='multi', add_param='', out_dir:str=None):
        """
        Function used to compute landmarks of videos and saves the results in out_dir

        input:
            files(list of .jpg .png .bmp .jpeg files)
            fdir(string that represents a folder containing the videos)
            vtype('multi' | 'single'    (number of people in the video))
            add_param(additional configurations for OpenFace API)
            out_dir(path where output is stored)

            Note: use only one between files & fdir

        output:
            dict( filename-->class_for_extracting_data_from_that_filename )
        """
        res = {}
        if files is None and fdir is None:
            raise Exception(M.EXE_PROCESS_VID)
        assert vtype in ('single', 'multi')

        # get path where outputs are saved
        if not out_dir:
            out_dir = self.out_dir
        else:
            out_dir = self._get_abs_path(out_dir)

        # how meny people there are
        if vtype=='multi':
            exe = 'FaceLandmarkVidMulti'
        else:
            exe = 'FeatureExtraction'

        # get files parameters
        formats = ('.avi', '.mp4')
        more=None
        if fdir:
            fdir = self._get_abs_path(fdir)
            files = os.listdir(fdir)
            files = self._non_in_cache(files, res, output_extractor.VidDataExtractor)
            files = [f'-f {f}' for f in files if f[-4:] in formats]
            paths = [f for f in os.listdir(fdir) if f[-4:] in formats]
            if (len(files)>100):
                more  = files[100:]
                files = files[:100]
            src = f"-inroot {fdir} {' '.join(files)}"
        else:
            files = [self._get_abs_path(f) for f in files]
            files = self._non_in_cache(files, res, output_extractor.VidDataExtractor)
            paths = [f.split('/')[-1] for f in files]
            src = ' '.join([f'-f {f}' for f in files if f[-4:] in formats])



        if len(src)<=6 and len(res)==0:
            # if src is short->no files were found.... error
            raise Exception(M.EXE_PROCESS_VID_FILES)
        elif len(src)>6:
            # execute command if some videos were not in cache
            cmd = f"{self.exe_path}/{exe} {src} -out_dir {out_dir} {add_param}"
            os.system(cmd)

        while more:
            files = more[:100]
            more  = files[100:]
            src = f"-inroot {fdir} {' '.join(files)}"
            cmd = f"{self.exe_path}/{exe} {src} -out_dir {out_dir} {add_param}"
            os.system(cmd)

        for p in paths:
            p_path = os.path.join(out_dir, p.split('.')[0])
            # remove folder of output images (not used after)
            if (os.path.exists(p_path+'_aligned')):
                os.system(f'rm -r {p_path}_aligned')
            if (os.path.exists(p_path+'.avi')):
                os.system(f'rm -r {p_path}.avi {p_path}_of_details.txt')

            # get result
            res[p] = output_extractor.VidDataExtractor(p_path, self)

        return res

    def _non_in_cache(self, files, res, CL):
        """foreach file in files: if a csv for that file exists: create a DataExtractor for it"""
        found = set()
        for f in files:
            fname = ( os.path.isabs(f) and f.split('/')[-1] ) or f
            partial_path = os.path.join(self.out_dir, fname.split('.')[0])
            if os.path.exists(partial_path+'.csv'):
                res[fname] = CL(partial_path, self)
                found.add(f)
        return list(set(files)-found)

    def _check_init_files(self, openface_path):
        """Check that the path given in input is valid"""

        # Check that openface_path is correct
        # in theory Openface/exe/FaceLandmarkImg should contain FaceLandmarkImg.cpp
        exe_path = os.path.join(openface_path, 'build/bin')
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
