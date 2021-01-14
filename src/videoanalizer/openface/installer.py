###########################
#
#  for an automated fresh install of OpenCv in linux environments
#
##########################

import os, re

def install_openface():
    oldpath =  os.getcwd()
    mypath = '/'.join(str(__file__).split('/')[:-1])
    os.chdir(mypath)

    install = input('(if you have already installed Openface, pass as argument the path to OpenFace folder when calling openface.OpenFaceAPI())\nDo you want to in stall OpenFace? [y|N]\nONLY FOR LINUX BASED SYSTEMS\n--> ')
    if (install.lower()=='y'):

        ## download openface
        if not os.path.exists('./OpenFace'):
            os.system('git clone https://github.com/TadasBaltrusaitis/OpenFace.git')
            os.chdir(mypath+'/OpenFace')

        if not os.path.exists('./OpenFace/lib/local/LandmarkDetector/model/patch_experts/cen_patches_0.50_of.dat'):
            os.system('bash ./download_models.sh')

        # # changing some lines of install.sh (allow opencv to decode videos)
        with open(f'{mypath}/OpenFace/install.sh', 'r') as f:
            text = f.read()

        change1 = "sudo apt-get -y install libtbb2 libjpeg-dev libpng-dev libtiff-dev libdc1394-22-dev\nsudo apt-get -y install ffmpeg\nsudo apt-get -y install libavcodec-dev libavformat-dev libavdevice-dev"
        text = re.sub('sudo apt-get -y install libtbb2 libjpeg-dev libpng-dev libtiff-dev libdc1394-22-dev', change1, text)

        change2 = "cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D WITH_FFMPEG=ON -D WITH_TBB=ON -D WITH_CUDA=OFF -D BUILD_SHARED_LIBS=OFF WITH_GSTREAMER=ON .."
        text = re.sub( "cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D WITH_TBB=ON -D WITH_CUDA=OFF -D BUILD_SHARED_LIBS=OFF ..", change2, text)

        with open(f'{mypath}/OpenFace/install.sh', 'w') as f:
            f.write(text)

        # install OpenFace
        os.chdir(mypath+'/OpenFace')
        os.system('sudo bash ./install.sh ')
        os.chdir(oldpath)
    else:
        print('installation stopped.\nopenface python module can\'t work without the compiled OpenFace project')
        exit(0)
