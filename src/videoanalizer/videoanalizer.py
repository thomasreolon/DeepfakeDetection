import os, random, numpy as np
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
    
    Attributes
    ----------

    config:  dict
        default configurations when processing a video
    
    api:     OpenFaceAPI
        instance of OpenFaceAPI, which is a wrapper module that interact with the C implementation of OpenFace
    """
    def __init__(self, **kw):
        """
        This function create an instance of VideoAnalizer, which exposes some methods to process videos

        Parameters
        ----------
        openface_dir='./openface/OpenFace': str
            path to the root folder of OpenFace, default: inside openface module

        interval=WHOLE VIDEO:               list of intervals
            which intervals of frames must be processed
        
        frames_per_sample=-1:               int
            how many frames are needed to generate a sample, default 1 sample per video

        overlap=0:                          int
            generate more samples in proportion to overlap, but they can overlap
        
        only_success=True:                  bool
            if True exclude frames where OpenFace was not confident

        vtype='single':                     'single'|'multi'
            which OpenFace executable is to be used. multi can track multiple faces, but it is less precise

        out_dir='../../output':             str
            where to save the results of the computations
        """
        openface_dir = 'openface_dir' in kw and kw['openface_dir'] or None
        openface_out_dir = openface_dir or '/'.join(str(__file__).split('/')[:-1])+'/../../output/openfacesaves'
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

    def process_video(self, files=None, fdir=None, config=None, rich_features=0):
        """
        returns a list of array where each array contains the 190 features.
        number of arrays = sum_video [ n_frames(get_only_frames_in(video, interval))/frames_per_sample ]
        config specifies how to analyze videos (eg. split them)

        we take the 20 Action Units from openface, than we look for the correlation between them (which is the 190 final features)

        Steps
        -----
        1) pass the video to openface
        2) get ActionUnits from openface
        3) split the videos in more parts (as specified in config, default no splits)
        4) get the correlation for each part
        5) return the correlations

        Parameters
        ----------

        files:                  list of strings
            list of files to process  (only one between files and fdir will be used)

        fdir:                   str
            folder containing videos to process 

        config:                 dict
            config that will override
            self.config = {
                'interval':[(0,1e20)],   #  intervals of frame to use, default: whole video
                'frames_per_sample': -1, #  -1 means every frame ; >0 means that a video with 600 frames will have 600/frames_per_sample samples
                'overlap': 0,            #  add overlapping samples (0->NO, 1->double, 2->triple)
                'only_success': True,    #  drop frames where openface is not confident
                'vtype': 'single',       #  detect one 'single' or multiple 'multi' faces in the videos
                'out_dir': '/'.join(str(__file__).split('/')[:-1])+'/../../output',
            }

        rich_features=0:        int
            0 --> 190 features as specified in Agarwal et al.
            2 --> 60 features wich are average, std and max of the 20 Action Units
            1 --> 250 features =  60 + 190

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
                if (rich_features == 1): # instead of 190 --> 250
                    features += get_rich_features(raw_features)
                if(rich_features == 2): # only reach features
                    features = get_rich_features(raw_features)

                if np.all(np.isfinite(features)):
                    samples.append(features)
                    videoids.append(fname)

        return samples, videoids



    def plot_features(self, folders_list, root_dir=None, labels=None, plot_type='PCA', config=None, save_path=None):
        """
        folders_list is a list of list of folders. each folder contains videos.
        each list of folders will have the same label.
        root_dir is the relative path to the folders.

        eg. plot_features(folders_list=[[obama1, obama2], [fakeobama]], rootdir='./videos', labels=['obama', 'fake'])
        
        videos
        |--obama1
            |----vid1.mp4
            |----vid2.mp4
        |--obama2
            |----vid0.mp4
        |--fakeobama
            |----fake.mp4
            
        Parameters
        ----------
        folders_list:        list of list
            each sublist contains paths to folders of videos. each sublist will have a different color in the final plot
        
        root_dir:             str
            root path,  folder list path will be calculated relatively from there

        labels:               list of string
            a label to assign to each sublist in folder_list. default: the first folder in the sublist

        plot_type:            'PCA'|'LDA'
            which feature extraction method to use

        config:               dict
            config that will override
            self.config = {
                'interval':[(0,1e20)],   #  intervals of frame to use, default: whole video
                'frames_per_sample': -1, #  -1 means every frame ; >0 means that a video with 600 frames will have 600/frames_per_sample samples
                'overlap': 0,            #  add overlapping samples (0->NO, 1->double, 2->triple)
                'only_success': True,    #  drop frames where openface is not confident
                'vtype': 'single',       #  detect one 'single' or multiple 'multi' faces in the videos
                'out_dir': '/'.join(str(__file__).split('/')[:-1])+'/../../output',
            }

        save_path:           str
            where to store the plot. If None the plot will be shown live (plt.show())
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

    def split_train_test(self, real_dir, fake_dir=None, train_fraction=0.66, deterministic = False, config=None, rich_features=0):
        """
        A function to split the dataset in train and test set

        Parameters
        ----------

        real_dir:       str
            path to a directory that cantains real videos (of the same person)
 
        fake_dir:       str
            path to a directory that contains fake videos (of the same person)

        train_fraction: float
            how many video assign to training and test in proportion

        deterministic:  bool
            if the split is random or deterministic

        config:               dict
            config that will override self.config
            self.config = {
                'interval':[(0,1e20)],   #  intervals of frame to use, default: whole video
                'frames_per_sample': -1, #  -1 means every frame ; >0 means that a video with 600 frames will have 600/frames_per_sample samples
                'overlap': 0,            #  add overlapping samples (0->NO, 1->double, 2->triple)
                'only_success': True,    #  drop frames where openface is not confident
                'vtype': 'single',       #  detect one 'single' or multiple 'multi' faces in the videos
                'out_dir': '/'.join(str(__file__).split('/')[:-1])+'/../../output',
            }

        rich_features=0:        int
            0 --> 190 features as specified in Agarwal et al.
            2 --> 60 features wich are average, std and max of the 20 Action Units
            1 --> 250 features =  60 + 190
        
        """

        # process video
        config = config or self.config
        X_r, vids = self.process_video(fdir=real_dir, config=config, rich_features=rich_features)

        # how many videos in the training set
        unique_vids = list(set(vids))
        k=1+int(len(unique_vids)*(train_fraction))

        # which videos in the training set
        train_vids_id = set(unique_vids[i] for i in range(k))  if deterministic else set(random.choices(unique_vids, k=k))

        # generate dataset
        x_train = [x for x,v_id in zip(X_r, vids) if v_id in train_vids_id]
        x_test  = [x for x,v_id in zip(X_r, vids) if v_id not in train_vids_id]
        y_train = [1]*len(x_train)
        y_test  = [1]*len(x_test)

        # do the same for fakes (binary classifier)
        if fake_dir:
            X_f, vids = self.process_video(fdir=fake_dir, config=config, rich_features=rich_features)
            unique_vids = list(set(vids))
            k=1+int(len(unique_vids)*(train_fraction))
            train_vids_id = set(unique_vids[i] for i in range(k))  if deterministic else set(random.choices(unique_vids, k=k))
            x_train += [x for x,v_id in zip(X_f, vids) if v_id in train_vids_id]
            x_test  += [x for x,v_id in zip(X_f, vids) if v_id not in train_vids_id]
            y_train += [-1]*len(x_train)
            y_test  += [-1]*len(x_test)

        return x_train, y_train, x_test, y_test

    def train_OneClassSVM(self, directory_of_videos, config=None, rich_features=0, boosted=True, person="unknown"):
        """
        Parameters
        ----------
        directory_of_videos:     str
            folder containing real people of the same person

        config:                  dict
            settings to extract samples
            default_config = {
                'interval':[(0,1e20)],   #  intervals of frame to use, default: whole video
                'frames_per_sample': -1, #  -1 means every frame ; >0 means that a video with 600 frames will have 600/frames_per_sample samples
                'overlap': 0,            #  add overlapping samples (0->NO, 1->double, 2->triple)
                'only_success': True,    #  drop frames where openface is not confident
                'vtype': 'single',       #  detect one 'single' or multiple 'multi' faces in the videos
                'out_dir': '/'.join(str(__file__).split('/')[:-1])+'/../../output',
            }

        rich_features:           int  
            ignored if boosted True: default parameters will be used, see classifier.py
            0 --> 190 features as specified in Agarwal et al.
            2 --> 60 features wich are average, std and max of the 20 Action Units
            1 --> 250 features =  60 + 190

        boosted:                 bool      
            if True it uses the pipeline of models (15) to train and predict, one OneClassSVM otherwise
        
        person:                  str
            the person to which the videos are related
        """
        config = self._get_config(config or {'frames_per_sample':300})
        # extract 250 features (X)
        X, _ = self.process_video(fdir=directory_of_videos, config=config, rich_features=1)
        # choose method based on boosted
        Clf =  (boosted and BoostedOneClassRbf) or OneClassRbf
        # initialize and train the model
        clf = Clf(self, rich_features=rich_features, person = person, config=config)
        clf.fit(X)

        return clf
