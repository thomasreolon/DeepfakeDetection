import os, pathlib
from videoanalizer import VideoAnalizer

'''
run download_dataset.py to get the videos before trying this script

Starting from a dataset of real and fake videos about people, this script splits these videos in test set (fake + some real videos) and training set (only real videos)
divided based on the video's person.
Then it trains the pipeline of models (one pipeline for each person of the dataset)
Then it tests the perfomances on the test set, the results will be stored in ../output/results/final_results.txt
'''

# Extract test set (fake)
a = [x for x in os.listdir('../test_data/videos/fake') if '__' not in x]
# thomas1: 0, thomas2 (friend): 1, moreno: 2, Elon: 3, Obama: 4, Renzi: 5
people = ["thomas1", "thomas2", "morez", "ElonMusk", "Obama", "Renzi"]
videos_test = []
for person in people:
    p = 0 if person == "thomas1" else 1 if person == "thomas2" else 2 if person == "morez" else 3 if person == "ElonMusk" else 4 if person == "Obama" else 5
    path = '../test_data/videos/fake/' + person
    for file in os.listdir(path):
        videos_test.append([path + "/" + file, p, "fake"])

#Add manually real videos for testing
#path, index of the person, true label
videos_test.append(['../test_data/videos/real/Obama/test/obama1.mp4',4, "real"])
videos_test.append(['../test_data/videos/real/Obama/test/obama1_real.mp4',4, "real"])
videos_test.append(['../test_data/videos/real/Obama/test/President_Barack_Obama_NCB.mp4',4, "real"])
videos_test.append(['../test_data/videos/real/Obama/test/The_Speech_that_Made_Obama_President.mp4',4, "real"])
videos_test.append(['../test_data/videos/real/Obama/test/Discorsi_obamaWG-0.mp4',4, "real"])

videos_test.append(['../test_data/videos/real/ElonMusk/test/elon4real.mp4',3, "real"])
videos_test.append(['../test_data/videos/real/ElonMusk/test/Elon_MuskpYH8.mp4',3, "real"])

# For debugging & consistency
os.chdir(pathlib.Path(__file__).parent.absolute())

clf = [] # clf[0] = thomas1, clf[1] = thomas2, clf[2] = moreno, clf[3] = Elon, clf[4] = Obama, clf[5] = Renzi

# Train a pipeline for each person
# Thomas
vd = VideoAnalizer()
# Boosted = True -> use the pipeline of models, the best method
clf.append(vd.train_OneClassSVM('../test_data/videos/real/thomas1', boosted=True, person = "thomas1"))
# Other Thomas (friend)
vd2 = VideoAnalizer()
clf.append(vd2.train_OneClassSVM('../test_data/videos/real/thomas2', boosted=True, person = "thomas2"))
# Moreno
vd3 = VideoAnalizer()
clf.append(vd3.train_OneClassSVM('../test_data/videos/real/morez', boosted=True, person = "morez"))
# Elon Musk
vd4 = VideoAnalizer()
clf.append(vd4.train_OneClassSVM('../test_data/videos/real/ElonMusk/train', boosted=True, person = "ElonMusk"))
# Obama
vd5 = VideoAnalizer()
clf.append(vd5.train_OneClassSVM('../test_data/videos/real/Obama/train', boosted=True, person = "Obama"))
# Renzi
vd6 = VideoAnalizer()
clf.append(vd6.train_OneClassSVM('../test_data/videos/real/Renzi', boosted=True, person = "Renzi"))

# Tests

results = {}

TP = 0 #True Positive
FP = 0 #False Positive
FN = 0 #False Negative

# For each test video, predict its label and store the results for the accuracies
for video in videos_test:
    result = clf[video[1]].predict_video(video[0], return_label=True)
    results[video[0]] = result
    TP += 1 if result == video[2] else 0
    FP += 1 if ((result == 'real') and (video[2] == 'fake')) else 0
    FN += 1 if ((result == 'fake') and (video[2] == 'real')) else 0

# Write the results
file = open("../output/final_results.txt", "w+")

file.write(f"Accuracy: {TP/len(videos_test)}\n") # Simple accuracy
file.write(f"f-1 score: {TP/(TP+(0.5*(FP+FN)))}\n") # f1 score
file.write("-------\n")

file.close()
