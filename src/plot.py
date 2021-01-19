from videoanalizer import VideoAnalizer


DATA_DIR = '../test_data/fake_vs_real'


vd = VideoAnalizer()


vd.plot_features(DATA_DIR, plot_type='PCA')
vd.plot_features(DATA_DIR, plot_type='LDA')

