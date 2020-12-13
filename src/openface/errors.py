###########################
# 
#  file containing error messages
#
##########################


############################## openfaceAPI init  ############################################
INIT_INVALID_PATH = """
The provided path:'{}', is not valid.
You should provide the absolute path where OpenFace was downloaded.
(eg. using the command:
    $ cd /home/user1 && git clone https://github.com/TadasBaltrusaitis/OpenFace.git
the path to provide will be:
    /home/user1/OpenFace
)
"""

INIT_EXE_NOT_FOUND = """
Are you sure to have run install.sh in OpenFace root folder?
After executing that command the folders /build/bin should have been created
and you should be able to see ['FaceLandmarkImg', 'FaceLandmarkVid', ...] at '{}'
"""

INIT_MODELS_NOT_FOUND = """
Are you sure to have run download_models.sh in OpenFace root folder?
After executiong that command you should have cen_patches_0.50_of.dat in OpenFace/lib/local/LandmarkDetector/model/patch_experts
"""

############################### openfaceAPI functions ####################################

PATH_NOT_FOUND = """
Could not find {}
"""

EXE_PROCESS_IMG = """
At least one file or one file_directory must be specified.
process_images(files=['a.jpg', 'b.png'])   or process_images(fdir='./images')
"""

EXE_PROCESS_VID = """
At least one file or one file_directory must be specified.
process_images(files=['a.mp4', 'b.avi'])   or process_images(fdir='./vids')
"""

EXE_PROCESS_VID_FILES = """
No valid files found. The only valid format is .avi
"""

EXE_LANDMARK_IMG = """
an error occoured while calling FaceLandmarkImg...
specific cmd: $ {}
ERROR: {}
"""


################################################################################