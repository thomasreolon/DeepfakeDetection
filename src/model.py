import os, pathlib, math
from sklearn.svm import LinearSVC
from videoanalizer import VideoAnalizer
from sklearn import metrics
from sklearn.neural_network import MLPClassifier

PERSON = 'Obama'

def filter_features(features):
    features_new = features[:]
    for index,video in enumerate(features):
        for f in video:
            if(math.isnan(f)):
                del features_new[index]
                break
    return features_new

path_training_fake = f'../test_data/videos/fake/{PERSON}/test'
path_training_real = f'../test_data/videos/real/{PERSON}/test'
path_test_fake = f'../test_data/videos/fake/{PERSON}/training'
path_test_real = f'../test_data/videos/real/{PERSON}/training'

vd = VideoAnalizer()

config = {'frames_per_sample':300}
features_training_real = vd.process_video(fdir=path_training_real, config=config)
features_training_fake = vd.process_video(fdir=path_training_fake, config=config)
features_test_real = vd.process_video(fdir=path_test_real, config=config)
features_test_fake = vd.process_video(fdir=path_test_fake, config=config)

features_training_real = filter_features(features_training_real)
features_training_fake = filter_features(features_training_fake)
features_test_real = filter_features(features_test_real)
features_test_fake = filter_features(features_test_fake)

class_training_real = ['real' for i in range(0,len(features_training_real))]
class_training_fake = ['fake' for i in range(0,len(features_training_fake))]
class_test_real = ['real' for i in range(0,len(features_test_real))]
class_test_fake = ['fake' for i in range(0,len(features_test_fake))]

training_features = features_training_real + features_training_fake
training_labels = class_training_real + class_training_fake

test_features = features_test_real + features_test_fake
test_labels = class_test_real + class_test_fake

MLP = LinearSVC()

MLP.fit(training_features, training_labels)

predicted = MLP.predict(test_features)

print(metrics.classification_report(test_labels, predicted))
