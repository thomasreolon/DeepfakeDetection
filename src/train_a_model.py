import os, pathlib
from videoanalizer import VideoAnalizer

# For debugging & consistency
os.chdir(pathlib.Path(__file__).parent.absolute()) 

# Train Classifier
vd = VideoAnalizer()
clf = vd.train_OneClassSVM('../test_data/videos/real/me')

# Test
result = clf.predict_video('../test_data/videos/fake/me/fake_me.mp4', return_label=True)
print(f'should be fake, predicted: {result}')
