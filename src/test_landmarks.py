import os, pathlib
from videoanalizer import VideoAnalizer

vd = VideoAnalizer()

clf = vd.train_OneClassSVM('../test_data/videos/real/thomas1')

print('startss')
clf.predict_video('../test_data/videos/fake/thomas1/fake_me.mp4', landmark_video=True)