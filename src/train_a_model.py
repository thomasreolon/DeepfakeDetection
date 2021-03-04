import os, pathlib
from videoanalizer import VideoAnalizer

# For debugging & consistency
os.chdir(pathlib.Path(__file__).parent.absolute()) 

# Train Classifier 1
vd = VideoAnalizer()
clf = vd.train_OneClassSVM('../test_data/videos/real/me')

# Test
result = clf.predict_video('../test_data/videos/fake/me/fake_me.mp4', return_label=True)
print(f'me-stallone should be fake, predicted: {result}')

# Train Classifier 2
vd = VideoAnalizer()
clf = vd.train_OneClassSVM('../test_data/videos/real/morez', boosted=False)

# Test
results=[]
result = clf.predict_video('../test_data/videos/fake/morez/1.mp4', return_label=True)
results.append(f'morez-redhead should be fake, predicted: {result}')
result = clf.predict_video('../test_data/videos/fake/morez/2.mp4', return_label=True)
results.append(f'morez-mylos should be fake, predicted: {result}')
result = clf.predict_video('../test_data/videos/fake/morez/3.mp4', return_label=True)
results.append(f'morez-superman should be fake, predicted: {result}')
result = clf.predict_video('../test_data/videos/real/morez/VID20210303133629.mp4', return_label=True)
results.append(f'morez talking should be real, predicted: {result}')

for r in results:
    print(r)
