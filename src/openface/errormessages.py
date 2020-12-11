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

EXE_LANDMARK_IMG = """
an error occoured while calling FaceLandmarkImg...
specific cmd: $ {}
ERROR: {}
"""


################################################################################