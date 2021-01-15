import videoanalizer.openface as openface
import videoanalizer.openface.parts as parts
from .samplesextractor import extract_samples
from .covariance import get_190_features

class VideoAnalizer():
    """
    This Class works as an interface to analyze videos and obtain the 190 features
    specified in the paper for fake detection.
    the video are analyzed calling openface api, which is a python API for OpenFace

    you can specify configs like interval (which frames),  frames per samples (divide the interval in more parts), ...
    """
    def __init__(self, **kw):
        openface_out_dir = '/'.join(str(__file__).split('/')[:-1])+'/../../output/openfacesaves'
        self.api = openface.OpenFaceAPI(out_dir=openface_out_dir)      # use openfaceAPI to process videos at low level
        # Default configs
        self.config = {
            'interval':[(0,1e20)],   #  intervals of frame to use, default: whole video
            'frames_per_sample': -1, #  -1 means every frame ; >0 means that a video with 600 frames will have 600/frames_per_sample samples
            'overlap': 0,            #  add overlapping samples (0->NO, 1->double, 2->triple)
            'only_success': True,    #  drop frames where openface is not confident
            'vtype': 'single',       #  detect one 'single' or multiple 'multi' faces in the videos
            'out_dir': '/'.join(str(__file__).split('/')[:-1])+'/../../output/features',
        }
        for k, v in kw.items():
            if k in self.config:
                self.config[k] = v


    def process_video(self, files=None, fdir=None, config=None):
        """
        returns a list of array where each array contains the 190 features.
        number of arrays = sum_video [ n_frames(get_only_frames_in(video, interval))/frames_per_sample ]
        config specifies how to analyze videos (eg. split them)

        we take the 20 Action Units from openface, than we look for the correlation between them (which is the 190 final features)

        steps:
        1) pass the video to openface
        2) get ActionUnits from openface
        3) split the videos in more parts (as specified in config, default no splits)
        4) get the correlation for each part
        5) return the correlations

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
            features = get_190_features(features)
            samples.append(features)

        return samples


