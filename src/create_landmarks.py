import os, pathlib, random, json
from videoanalizer import VideoAnalizer

os.chdir(pathlib.Path(__file__).parent.absolute())
vd = VideoAnalizer()

########### Video with Landmarks

# Train the model
clf = vd.train_OneClassSVM('../test_data/videos/real/Obama/train', boosted=True)
# Predict and save landmarks
clf.predict_video('../test_data/videos/real/Obama/test/obama1_real.mp4', landmark_video=True)
clf.predict_video('../test_data/videos/fake/Obama/Obama_fake_5.mp4', landmark_video=True)

# Repeat for Elon
clf2 = vd.train_OneClassSVM('../test_data/videos/real/ElonMusk/train', boosted=True)
clf2.predict_video('../test_data/videos/real/ElonMusk/test/elon4real.mp4', landmark_video=True)
clf2.predict_video('../test_data/videos/fake/ElonMusk/ElonMusk_fake_3.mp4', landmark_video=True)
