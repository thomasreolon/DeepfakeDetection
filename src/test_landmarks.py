import os, pathlib
from videoanalizer import VideoAnalizer

vd = VideoAnalizer()


# Video with Landmarks

clf = vd.train_OneClassSVM('../test_data/videos/real/thomas1')
print('startss')
clf.predict_video('../test_data/videos/fake/thomas1/fake_me.mp4', landmark_video=True)


# Why are classifiers misbehaving?
vd.plot_features([['../test_data/videos/real/Obama'], ['../test_data/videos/fake/Obama']], labels=['real', 'fake']) 
