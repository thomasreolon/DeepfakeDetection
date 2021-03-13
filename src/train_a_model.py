import os, pathlib
from videoanalizer import VideoAnalizer

# 0 = thomas1, 1 = thomas2, 2 = moreno
a = [x for x in os.listdir('../test_data/videos/fake') if '__' not in x]
people = ["thomas1", "thomas2", "morez", "ElonMusk", "Obama", "Renzi"]
videos_test = []
for person in people:
    p = 0 if person == "thomas1" else 1 if person == "thomas2" else 2 if person == "morez" else 3 if person == "ElonMusk" else 4 if person == "Obama" else 5
    path = '../test_data/videos/fake/' + person
    for file in os.listdir(path):
        videos_test.append([path + "/" + file, p, "fake"])

videos_test.append(['../test_data/videos/real/Obama/test/obama1.mp4',4, "real"])
videos_test.append(['../test_data/videos/real/Obama/test/obama1_real.mp4',4, "real"])
videos_test.append(['../test_data/videos/real/Obama/test/President_Barack_Obama_NCB.mp4',4, "real"])
videos_test.append(['../test_data/videos/real/Obama/test/The_Speech_that_Made_Obama_President.mp4',4, "real"])
videos_test.append(['../test_data/videos/real/Obama/test/Discorsi_obamaWG-0.mp4',4, "real"])

videos_test.append(['../test_data/videos/real/ElonMusk/test/elon4real.mp4',3, "real"])
videos_test.append(['../test_data/videos/real/ElonMusk/test/Elon_MuskpYH8.mp4',3, "real"])

# For debugging & consistency
os.chdir(pathlib.Path(__file__).parent.absolute())

file = open("../output/results/final_results.txt", "w+")

for boosted in [True]:
    # RICH = [0]
    # if boosted:
    RICH = [0]
    for rich in RICH:
        R = "190" if rich==0 else "250" if rich==1 else "only rich features"
        clf = [] # clf[0] = thomas1, clf[1] = thomas2, clf[2] = moreno

        # Train Classifier 1 (thomas reolon)
        vd = VideoAnalizer()
        clf.append(vd.train_OneClassSVM('../test_data/videos/real/thomas1', rich_features=rich, boosted=boosted, person = "thomas1"))
        # Train Classifier 2 (other thomas)
        vd2 = VideoAnalizer()
        clf.append(vd2.train_OneClassSVM('../test_data/videos/real/thomas2', rich_features=rich, boosted=boosted, person = "thomas2"))
        # Train Classifier 3 (moreno)
        vd3 = VideoAnalizer()
        clf.append(vd3.train_OneClassSVM('../test_data/videos/real/morez', rich_features=rich, boosted=boosted, person = "morez"))

        vd4 = VideoAnalizer()
        clf.append(vd4.train_OneClassSVM('../test_data/videos/real/ElonMusk/train', rich_features=rich, boosted=boosted, person = "ElonMusk"))

        vd5 = VideoAnalizer()
        clf.append(vd5.train_OneClassSVM('../test_data/videos/real/Obama/train', rich_features=rich, boosted=boosted, person = "Obama", gridsearch = False))

        vd6 = VideoAnalizer()
        clf.append(vd6.train_OneClassSVM('../test_data/videos/real/Renzi', rich_features=rich, boosted=boosted, person = "Renzi", gridsearch = False))

        # Tests

        results = {}

        sum = 0
        for video in videos_test:
            # tl = 1 if video[2]=="real" else -1
            result = clf[video[1]].predict_video(video[0], return_label=True)
            results[video[0]] = result
            sum += 1 if result == video[2] else 0

        file.write(f"Boosted: {boosted} ")
        file.write(f"Rich: {R} ")
        file.write(f"Accuracy: {sum/len(videos_test)}\n")
        file.write("-------\n")
        # for c in clf:
        #     cms, person = c.get_confusion_matrix()
        #     file.write(f"Person: {person}: \n")
        #     for i, cm in enumerate(cms):
        #         file.write(f"{i}: {cm}\n")
        #     file.write("-------\n")

file.close()
