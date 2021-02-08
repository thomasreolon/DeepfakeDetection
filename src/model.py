import os, pathlib, math
from sklearn.svm import LinearSVC
from videoanalizer import VideoAnalizer
from sklearn import metrics

PERSON = 'Obama'
TEST_PERC = 0.3

def filter_features(features):
    features_new = features[:]
    for index,video in enumerate(features):
        for f in video:
            if(math.isnan(f)):
                del features_new[index]
                break
    return features_new

path_fake = f'../test_data/videos/fake/{PERSON}'
path_real = f'../test_data/videos/real/{PERSON}'

vd = VideoAnalizer()

config = {'frames_per_sample':300}
features_real = vd.process_video(fdir=path_real, config=config)
features_fake = vd.process_video(fdir=path_fake, config=config)

features_real = filter_features(features_real)
features_fake = filter_features(features_fake)

class_real = ['real' for i in range(0,len(features_real))]
class_fake = ['fake' for i in range(0,len(features_fake))]

slice_real = int(len(features_real)*(1-TEST_PERC))
slice_fake = int(len(features_fake)*(1-TEST_PERC))

training_features_real = features_real[0:slice_real]
training_labels_real = class_real[0:slice_real]
training_features_fake = features_fake[0:slice_fake]
training_labels_fake = class_fake[0:slice_fake]

test_features_real = features_real[slice_real:]
test_labels_real = class_real[slice_real:]
test_features_fake = features_fake[slice_fake:]
test_labels_fake = class_fake[slice_fake:]

training_features = training_features_real + training_features_fake
training_labels = training_labels_real + training_labels_fake

test_features = test_features_real + test_features_fake
test_labels = test_labels_real + test_labels_fake

svc = LinearSVC()

print(training_features)

svc.fit(training_features, training_labels)

predicted = svc.predict(test_features)

print(metrics.classification_report(test_labels, predicted))
