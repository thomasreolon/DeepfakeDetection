import os, pathlib
from videoanalizer import VideoAnalizer

# 0 = thomas1, 1 = thomas2, 2 = moreno
# videos_test =  [['../test_data/videos/fake/thomas1/fake_me.mp4',0],
#                 ['../test_data/videos/fake/thomas1/video_1.mp4',0],
#                 ['../test_data/videos/fake/thomas1/video_2.mp4',0],
#                 ['../test_data/videos/fake/thomas1/video_3.mp4',0],
#                 #['../test_data/videos/fake/thomas1/video_4.mp4',0], it generates 0 samples...
#                 ['../test_data/videos/fake/thomas2/video_1.mp4',1],
#                 ['../test_data/videos/fake/thomas2/video_2.mp4',1],
#                 # ['../test_data/videos/fake/morez/1.mp4',2], it generates 0 samples...
#                 ['../test_data/videos/fake/morez/2.mp4',2],
#                 ['../test_data/videos/fake/morez/3.mp4',2],
#                 ['../test_data/videos/fake/ElonMusk/elon2.mp4',3],
#                 ['../test_data/videos/fake/ElonMusk/elon3.mp4',3],
#                 ['../test_data/videos/fake/ElonMusk/elon4.mp4',3],
#                 ['../test_data/videos/fake/ElonMusk/Elon_Fakezlcc.mp4',3],
#                 ['../test_data/videos/fake/ElonMusk/musk_fake.mp4',3],
#                 ['../test_data/videos/fake/ElonMusk/odissey.mp4',3],
#                 ['../test_data/videos/fake/ElonMusk/reface-2021-02-16-01-35-35.mp4',3],
#                 ['../test_data/videos/fake/ElonMusk/reface-2021-02-16-01-36-16.mp4',3],
#                 ['../test_data/videos/fake/ElonMusk/reface-2021-02-16-01-37-29.mp4',3],
#                 ['../test_data/videos/fake/ElonMusk/reface-2021-02-16-01-38-36.mp4',3],
#                 ['../test_data/videos/fake/ElonMusk/reface-2021-02-16-01-39-32.mp4',3],
#                 ['../test_data/videos/fake/ElonMusk/reface-2021-02-16-01-42-40.mp4',3],
#                 ['../test_data/videos/fake/Obama/Barack_Obama_KellyDeepfake4.mp4',4],
#                 ['../test_data/videos/fake/Obama/Barack_Obama_Kelly_Deepfake.mp4',4],
#                 ['../test_data/videos/fake/Obama/Barack_Obama_Kelly_Deepfake2.mp4',4],
#                 ['../test_data/videos/fake/Obama/Barack_Obama_Kelly_Deepfake3.mp4',4],
#                 ['../test_data/videos/fake/Obama/fakevideo_Obama.mp4',4],
#                 ['../test_data/videos/fake/Obama/obamafake.mp4',4],
#                 ['../test_data/videos/fake/Obama/Obama_Fake1eL0.mp4',4],
#                 ['../test_data/videos/fake/Obama/obamaMit.mp4',4],
#                 ['../test_data/videos/fake/Obama/Obama_Sings_Spooky_Scary.mp4',4],
#                 ['../test_data/videos/fake/Obama/Obama_Sings_Spooky_Scary2.mp4',4],
#                 ['../test_data/videos/fake/Obama/VID-20210208-WA0003.mp4',4],
#                 ['../test_data/videos/fake/Obama/VID-20210208-WA0004.mp4',4],
#                 ['../test_data/videos/fake/Obama/VID-20210208-WA0005.mp4',4],
#                 ['../test_data/videos/fake/Obama/VID-20210208-WA0006.mp4',4],
#                 ['../test_data/videos/fake/Obama/videoplayback.mp4',4],
#                 ['../test_data/videos/fake/Obama/videoplayback2.mp4',4]]

# videos_test =  [['../test_data/videos/real/ElonMusk/test/elon4real.mp4',0],
#                 ['../test_data/videos/real/ElonMusk/test/Elon_MuskpYH8.mp4',0],
#                 ['../test_data/videos/real/Obama/test/Discorsi_obamaWG-0.mp4',1],
#                 ['../test_data/videos/real/Obama/test/obama1.mp4',1],
#                 ['../test_data/videos/real/Obama/test/obama1_real.mp4',1],
#                 ['../test_data/videos/real/Obama/test/President_Barack_Obama_NCB.mp4',1],
#                 ['../test_data/videos/real/Obama/test/The_Speech_that_Made_Obama_President.mp4',1]]

videos_test =  [['../test_data/videos/real/Obama/test/Discorsi_obamaWG-0.mp4',0],
                ['../test_data/videos/real/Obama/test/obama1.mp4',0],
                ['../test_data/videos/real/Obama/test/obama1_real.mp4',0],
                ['../test_data/videos/real/Obama/test/President_Barack_Obama_NCB.mp4',0],
                ['../test_data/videos/real/Obama/test/The_Speech_that_Made_Obama_President.mp4',0]]


# For debugging & consistency
os.chdir(pathlib.Path(__file__).parent.absolute())

file = open("../output/results/final_results.txt", "w+")

for boosted in [False]:
    # RICH = [0]
    # if boosted:
    RICH = [0]
    for rich in RICH:
        R = "190" if rich==0 else "250" if rich==1 else "only rich features"
        clf = [] # clf[0] = thomas1, clf[1] = thomas2, clf[2] = moreno

        # Train Classifier 1 (thomas reolon)
        # vd = VideoAnalizer()
        # clf.append(vd.train_OneClassSVM('../test_data/videos/real/thomas1', rich_features=rich, boosted=boosted, person = "thomas1"))
        # # Train Classifier 2 (other thomas)
        # vd2 = VideoAnalizer()
        # clf.append(vd2.train_OneClassSVM('../test_data/videos/real/thomas2', rich_features=rich, boosted=boosted, person = "thomas2"))
        # # Train Classifier 3 (moreno)
        # vd3 = VideoAnalizer()
        # clf.append(vd3.train_OneClassSVM('../test_data/videos/real/morez', rich_features=rich, boosted=boosted, person = "morez"))

        # vd = VideoAnalizer()
        # clf.append(vd.train_OneClassSVM('../test_data/videos/real/ElonMusk/train', rich_features=rich, boosted=boosted, person = "ElonMusk"))

        vd = VideoAnalizer()
        clf.append(vd.train_OneClassSVM('../test_data/videos/real/Obama/train', rich_features=rich, boosted=boosted, person = "Obama", gridsearch = True))

        # Tests

        results = {}

        for video in videos_test:
            result = clf[video[1]].predict_video(video[0], return_label=True)
            results[video[0]] = result

        file.write(f"Boosted: {boosted}\n")
        file.write(f"Rich: {R}\n")
        for r in results:
            file.write(f"---> Video: {r}\n")
            file.write(f"predictions:\n")
            for i in results[r]:
                file.write(f"             {i}\n")
        file.write("\n-------\n\n")

file.close()
