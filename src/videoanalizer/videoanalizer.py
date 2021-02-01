import os
from joblib import dump
from .openface import OpenFaceAPI
from .openface import parts
from .samplesextractor import extract_samples
from .covariance import get_190_features
from .classifier import train_specific_person_classifier
from .plots import plot_features2D

class VideoAnalizer():
    """
    This Class works as an interface to analyze videos and obtain the 190 features
    specified in the paper for fake detection.
    the video are analyzed calling openface api, which is a python API for OpenFace

    you can specify configs like interval (which frames),  frames per samples (divide the interval in more parts), ...
    """
    def __init__(self, **kw):
        openface_out_dir = '/'.join(str(__file__).split('/')[:-1])+'/../../output/openfacesaves'
        self.api = OpenFaceAPI(out_dir=openface_out_dir)      # use openfaceAPI to process videos at low level
        # Default configs
        self.config = {
            'interval':[(0,1e20)],   #  intervals of frame to use, default: whole video
            'frames_per_sample': -1, #  -1 means every frame ; >0 means that a video with 600 frames will have 600/frames_per_sample samples
            'overlap': 0,            #  add overlapping samples (0->NO, 1->double, 2->triple)
            'only_success': True,    #  drop frames where openface is not confident
            'vtype': 'single',       #  detect one 'single' or multiple 'multi' faces in the videos
            'out_dir': '/'.join(str(__file__).split('/')[:-1])+'/../../output',
        }
        self.config = self._get_config(kw)

    def _get_config(self, config):
        tmp = {**self.config}
        if (isinstance(config, dict)):
            for k, v in config.items():
                if k in tmp:
                    tmp[k] = v
        return tmp

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

        config = self._get_config(config)

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


    def train_classifier(self, person_files, non_person_files, person_name='Real', clf_type='OneClassSVM', config=None, show_trainig_performance=False, save=False):
        """
        input:
            person_files is a list of folders or files that contain a person
            person_non_files is a list folders or files that contains fakes, other people
            person_name is the name of the person in person_files

        output:
            a classifier that can recognize person_name in videos
        """
        config = self._get_config(config or {'frames_per_sample':300})
        samples = [[], []]
        for i,f in enumerate((person_files, non_person_files)):

            if (isinstance(f, str)):
                f = [f]

            files = []
            for path in f:
                if os.path.isdir(path):
                    samples[i] += self.process_video(fdir=path, config=config)
                else:
                    files.append(path)
            if files:
                samples[i] += self.process_video(files=files, config=config)
        
        clf = train_specific_person_classifier(samples[0], samples[1], self, person_name,clf_type, show_trainig_performance)
        if (save): self.save_classifier(clf)
        return clf

    def save_classifier(self,clf, fname=None, out_dir=None):
        out_dir = out_dir or self.config['out_dir']
        path = out_dir + f'/{fname or clf.labels_map[1]}-clf.joblib'
        dump(clf, path)


    def plot_features(self, folders_list, root_dir=None, labels=None, plot_type='PCA', config=None):
        """
        folders_list is a list of list of folders. each folder contains videos. 
        each list of folders will have the same label.
        root_dir is the relative path to the folders.

        videos
        |--obama1
            |----vid1.mp4
            |----vid2.mp4
        |--obama2
            |----vid0.mp4
        |--fakeobama
            |----fake.mp4


        plot_features(folders_list=[[obama1, obama2], [fakeobama]], rootdir='./videos', labels=['obama', 'fake'])

        """
        config = self._get_config(config or {'frames_per_sample':300})
        samples, fold_names = [], []
        fold = 'unlabeled'
        root_dir = root_dir and (root_dir+'/') or ''

        for folders in folders_list:
            tmp=[]
            for fold in folders:
                fdir = f'{root_dir}{fold}'
                tmp += self.process_video(fdir=fdir, config=config)
            samples.append(tmp)
            fold_names.append(fold)
            
        out_dir = self.config['out_dir']

        plot_features2D(samples, out_dir, labels or fold_names, plot_type, )

        


