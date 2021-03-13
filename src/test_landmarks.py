import os, pathlib, random, json
from videoanalizer import VideoAnalizer

vd = VideoAnalizer()
vd2 = VideoAnalizer()

# Video with Landmarks

clf = vd.train_OneClassSVM('../test_data/videos/real/Obama/train', boosted=True)
clf.predict_video('../test_data/videos/real/Obama/test/obama1_real.mp4', landmark_video=True)
clf.predict_video('../test_data/videos/fake/Obama/Obama_fake_5.mp4', landmark_video=True)

clf2 = vd2.train_OneClassSVM('../test_data/videos/real/ElonMusk/train', boosted=True)
clf2.predict_video('../test_data/videos/real/ElonMusk/test/elon4real.mp4', landmark_video=True)
clf2.predict_video('../test_data/videos/fake/ElonMusk/ElonMusk_fake_3.mp4', landmark_video=True)

quit()

# Why are classifiers misbehaving?
vd.plot_features([['../test_data/videos/real/Obama'], ['../test_data/videos/fake/Obama']], labels=['real', 'fake'])

# reals = [x for x in os.listdir('../test_data/videos/real') if '__' not in x]
# fakes = [x for x in os.listdir('../test_data/videos/fake') if '__' not in x]
#
# for person in reals:
#     path = '../test_data/videos/real/' + person
#     for index, file in enumerate(os.listdir(path)):
#         file_path = path + "/" + file
#         new_path = path + "/" + person + "_real_" + str(index) + ".mp4"
#         os.system(f"mv {file_path} {new_path}")
#
# for person in fakes:
#     path = '../test_data/videos/fake/' + person
#     for index, file in enumerate(os.listdir(path)):
#         file_path = path + "/" + file
#         new_path = path + "/" + person + "_fake_" + str(index) + ".mp4"
#         os.system(f"mv {file_path} {new_path}")
