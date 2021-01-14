import videoanalizer.openface as openface
import videoanalizer.openface.parts as parts
import numpy as np
from .samplesextractor import extract_samples
from .covariance import get_covariance

class VideoAnalizer():
    def __init__(self, config=None):
        self.api = openface.OpenFaceAPI()      # use openfaceAPI to process videos at low level
        self.config = config or {
            'interval':[(0,1e20)],   #  intervals of frame to use, default: whole video
            'frames_per_sample': -1, #  -1 means every frame ; >0 means that a video with 600 frames will have 600/frames_per_sample samples
            'overlap': 0.,           #  probability to have an overlapping frame
            'only_success': True,    #  drop frames where openface is not confident
            'vtype': 'single'        #  detect one 'single' or multiple 'multi' faces in the videos
        }

    def process_single_video(self, files=None, fdir=None, config=None):
        """
        returns a list of array where each array contains the 190 features.
        number of array = sum_video [ n_frames(get_only_frames_in(video, interval))/frames_per_sample ]
        """
        if files is None and fdir is None:
            raise Exception('at least one between files (eg. ["../obama.mp4"]) or fdir (eg. "../videos") must be specified')

        if config is None:
            config = self.config

        # process the video using openface
        res = self.api.process_video(files=files, fdir=fdir, vtype=config['vtype'])

        # divide the video into samples
        dx_samples = []
        for _, dx in res.items():
            dx_samples += extract_samples(dx, config)

        # get the 20 features used in the paper for each sample
        samples = []
        for s_dx in dx_samples:
            features = s_dx.get_raw_features(AU_r = parts.AU_paper_r,
                                  AU_c = parts.AU_paper_c,
                                  pose = parts.POSE_ROTATION_X_Z,
                                  mouth_h = parts.MOUTH_H,
                                  mouth_v = parts.MOUTH_V)
            features = get_covariance(features)
            samples.append(features)

        return samples


    def get_paper_input(self, of_dataextractor, mode='full_video'):
        """get the a list of arrays that contains the 190 correletion features between the features used in the paper
            input:
            - mode:  full_video | 10s | 10s_overlapping
        """
        return

        video = of_dataextractor.get_only_success()   # drop frames that can contain errors


            avgs = data.avg(dim=0)
            for i in range(len(data)):
                for j in range(i+1, len(data)):
                    f_array.append(00000)
            res.append(f_array)
        return array
