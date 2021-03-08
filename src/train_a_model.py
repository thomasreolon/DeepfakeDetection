import os, pathlib
from videoanalizer import VideoAnalizer

# 0 = thomas1, 1 = thomas2, 2 = moreno
videos_test =  [['../test_data/videos/fake/thomas1/fake_me.mp4',0],
                ['../test_data/videos/fake/thomas1/video_1.mp4',0],
                ['../test_data/videos/fake/thomas1/video_2.mp4',0],
                ['../test_data/videos/fake/thomas1/video_3.mp4',0],
                #['../test_data/videos/fake/thomas1/video_4.mp4',0], it generates 0 samples...
                ['../test_data/videos/fake/thomas2/video_1.mp4',1],
                ['../test_data/videos/fake/thomas2/video_2.mp4',1],
                # ['../test_data/videos/fake/morez/1.mp4',2], it generates 0 samples...
                ['../test_data/videos/fake/morez/2.mp4',2],
                ['../test_data/videos/fake/morez/3.mp4',2]]

# For debugging & consistency
os.chdir(pathlib.Path(__file__).parent.absolute())

file = open("../output/results/final_results.txt", "w+")

for boosted in [False, True]:
    # RICH = [0]
    # if boosted:
    RICH = [0,1,2]
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

        # Tests

        results = {}

        for video in videos_test:
            result = clf[video[1]].predict_video(video[0], return_label=True)
            results[video[0]] = result

        file.write(f"Boosted: {boosted}\n")
        file.write(f"Rich: {R}\n")
        for r in results:
            file.write(f"---> Video: {r}, predicted: {results[r]}\n")
        file.write("\n-------\n\n")

file.close()
