import os, random, numpy as np
from joblib import dump
from .openface import OpenFaceAPI
from .openface import parts
from .samplesextractor import extract_samples
from .covariance import get_190_features, get_rich_features
from .classifier import OneClassRbf, BoostedOneClassRbf
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

    def process_video(self, files=None, fdir=None, config=None, rich=False, rich_features=0):
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

        # divide the videos into samples, then extract the features from each sample
        samples, videoids = [], []
        for fname, video in res.items():
            for interval_of_video in extract_samples(video, config):
                raw_features = interval_of_video.get_raw_features(AU_r = parts.AU_paper_r,
                                    AU_c = parts.AU_paper_c,
                                    pose = parts.POSE_ROTATION_X_Z,
                                    mouth_h = parts.MOUTH_H,
                                    mouth_v = parts.MOUTH_V)
                features = get_190_features(raw_features)
                if (rich or (rich_features == 1)): # instead of 190 --> 250
                    features += get_rich_features(raw_features)
                if(rich_features == 2): # only reach features
                    features = get_rich_features(raw_features)

                if np.all(np.isfinite(features)):
                    samples.append(features)
                    videoids.append(fname)

        return samples, videoids


    def save_classifier(self,clf, fname=None, out_dir=None):
        out_dir = out_dir or self.config['out_dir']
        path = out_dir + f'/{fname or clf.labels_map[1]}-clf.joblib'
        dump(clf, path)


    def plot_features(self, folders_list, root_dir=None, labels=None, plot_type='PCA', config=None, save_path=None):
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
                tmp2, _ = self.process_video(fdir=fdir, config=config)
                tmp += tmp2
            samples.append(tmp)
            fold_names.append(fold)

        out_dir = save_path or self.config['out_dir']

        plot_features2D(samples, out_dir, labels or fold_names, plot_type, )

    def split_train_test(self, X, vid, train_fraction=0.66, labels_offset=None, deterministic = False):
        train_X, test_X, vids = [], [], sorted(list(set(vid)))
        k=1+int(len(vids)*(train_fraction))
        if(deterministic):
            train_vids_id = set(vids[i] for i in range(k))
        else:
            train_vids_id = set(random.choices(vids, k=k))
        labels = {'train':{}, 'test':{}}
        if labels_offset is None:
            labels_offset = (0,0)
        for x, d in zip(X, vid):
            if d in train_vids_id:
                labels['train'][len(train_X)+labels_offset[0]] = d
                train_X.append(x)
            else:
                labels['test'][len(test_X)+labels_offset[1]] = d
                test_X.append(x)
        return train_X, test_X, labels

    def train_OneClassSVM(self, directory_of_videos, config=None, rich_features=0, boosted=False, person="thomas1"):
        """
        input:
            - directory_of_videos:  folder containing real people of the same person
            - config:               settings to extract samples
            - rich_features:        use 190 features from the paper or 250
        """
        config = self._get_config(config or {'frames_per_sample':1000})
        X, _ = self.process_video(fdir=directory_of_videos, config=config, rich_features=rich_features)
        Clf =  (boosted and BoostedOneClassRbf) or OneClassRbf
        clf = Clf(self, rich_features, person = person)
        clf.fit(X)

        return clf
