import os, pathlib, random, json
from videoanalizer import VideoAnalizer

vd = VideoAnalizer()


# Video with Landmarks

clf = vd.train_OneClassSVM('../test_data/videos/real/thomas1', boosted=True)
clf.predict_video('../test_data/videos/real/thomas1/thomas1_real_0.mp4', landmark_video=True)
clf.predict_video('../test_data/videos/fake/thomas1/thomas1_fake_0.mp4', landmark_video=True)
clf.predict_video('../test_data/videos/fake/thomas1/thomas1_fake_1.mp4', landmark_video=True)

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
