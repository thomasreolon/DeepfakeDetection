import os, pathlib
from videoanalizer import VideoAnalizer

os.chdir(pathlib.Path(__file__).parent.absolute())
vd = VideoAnalizer()
ROOT_DIR = '../test_data/videos'
SAVE_PATH= '../output/plots'

if(not os.path.exists(SAVE_PATH)):
    os.mkdir(SAVE_PATH)

# Plot real videos from YT vs fake videos from YT
folders_list=[
    ['real/other'],
    ['fake/other']
]
labels = ['Real YT Videos', 'Fake YT Videos']
vd.plot_features(folders_list=folders_list, root_dir=ROOT_DIR, labels=labels, plot_type='PCA', save_path=SAVE_PATH)


# Plot real videos from YT vs fake videos from YT
folders_list=[
    ['real/other'],
    ['fake/other']
]
labels = ['Real YT Videos', 'Fake YT Videos']
vd.plot_features(folders_list=folders_list, root_dir=ROOT_DIR, labels=labels, plot_type='LDA', save_path=SAVE_PATH)
vd.plot_features(folders_list=folders_list, root_dir=ROOT_DIR, labels=labels, plot_type='PCA', save_path=SAVE_PATH)


# Plot Obama real videos vs Obama fake videos  (LDA)
folders_list=[
    ['real/Obama'],
    ['fake/Obama'],
]
labels = ['Real Obama Videos', 'Fake Obama Videos']
vd.plot_features(folders_list=folders_list, root_dir=ROOT_DIR, labels=labels, plot_type='PCA', save_path=SAVE_PATH)



# Plot Elon Musk real videos vs Elon Musk fake videos  (PCA)
folders_list=[
    ['real/ElonMusk'],
    ['fake/ElonMusk'],
]
labels = ['Real Elon Musk Videos', 'Fake Elon Musk Videos']
vd.plot_features(folders_list=folders_list, root_dir=ROOT_DIR, labels=labels, plot_type='PCA', save_path=SAVE_PATH)
vd.plot_features(folders_list=folders_list, root_dir=ROOT_DIR, labels=labels, plot_type='LDA', save_path=SAVE_PATH)


# Plot Elon real/fake videos vs Obama's (LDA)
folders_list=[
    ['real/ElonMusk'],
    ['fake/ElonMusk'],
    ['real/Obama'],
    ['fake/Obama'],
]
labels = ['Real Elon Musk Videos', 'Fake Elon Musk Videos', 'Real Obama Videos', 'Fake Obama Videos']
vd.plot_features(folders_list=folders_list, root_dir=ROOT_DIR, labels=labels, plot_type='LDA', save_path=SAVE_PATH)



# real fake features in 2D for many subjects
folders_list=[
    ['real/ElonMusk'],
    ['fake/ElonMusk'],
    ['real/Obama'],
    ['fake/Obama'],
    ['real/Mattarella'],
    ['fake/Mattarella'],
    ['real/Renzi'],
    ['fake/Renzi'],
    ['real/QueenElisabeth'],
    ['fake/QueenElisabeth'],
]
vd.plot_features(folders_list=folders_list, root_dir=ROOT_DIR)
vd.plot_features(folders_list=folders_list, root_dir=ROOT_DIR, plot_type='LDA', save_path=SAVE_PATH)


# real fake features in 2D for many subjects
folders_list=[
    ['real/Mattarella', 'real/ElonMusk', 'real/Obama', 'real/Renzi', 'real/QueenElisabeth'],
    ['fake/Mattarella', 'fake/Renzi', 'fake/Obama', 'fake/ElonMusk', 'fake/QueenElisabeth'],
]
vd.plot_features(folders_list=folders_list, root_dir=ROOT_DIR)
vd.plot_features(folders_list=folders_list, root_dir=ROOT_DIR, plot_type='LDA', save_path=SAVE_PATH)



# Plot challenge real videos vs challenge fake videos
folders_list=[
    ['real/__challenge/'+x for x in os.listdir(f'{ROOT_DIR}/real/__challenge')],
    ['fake/__challenge/'+x for x in os.listdir(f'{ROOT_DIR}/real/__challenge')],
]
labels = ['Real Challenge Videos', 'Fake Challenge Videos']
vd.plot_features(folders_list=folders_list, root_dir=ROOT_DIR, labels=labels, plot_type='LDA', save_path=SAVE_PATH)
vd.plot_features(folders_list=folders_list, root_dir=ROOT_DIR, labels=labels, plot_type='PCA', save_path=SAVE_PATH)


print("THE END!")