from videoanalizer import VideoAnalizer


vd = VideoAnalizer()

samples = vd.process_video(files=['../test_data/vid/obama2.mp4'])

print(samples)