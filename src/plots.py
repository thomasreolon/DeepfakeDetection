import os, pathlib
from videoanalizer import VideoAnalizer

os.chdir(pathlib.Path(__file__).parent.absolute())
vd = VideoAnalizer()
root_dir = '../test_data/videos'

# Plot real videos from YT vs fake videos from YT
folders_list=[
    ['real/other'],
    ['fake/other']
]
labels = ['Real YT Videos', 'Fake YT Videos']
vd.plot_features(folders_list=folders_list, root_dir=root_dir, labels=labels, plot_type='PCA')

exit(0)

# Plot real videos from YT vs fake videos from YT
folders_list=[
    ['real/other'],
    ['fake/other']
]
labels = ['Real YT Videos', 'Fake YT Videos']
vd.plot_features(folders_list=folders_list, root_dir=root_dir, labels=labels, plot_type='LDA')
vd.plot_features(folders_list=folders_list, root_dir=root_dir, labels=labels, plot_type='PCA')


# Plot Obama real videos vs Obama fake videos  (LDA)
folders_list=[
    ['real/Obama'],
    ['fake/Obama'],
]
labels = ['Real Obama Videos', 'Fake Obama Videos']
vd.plot_features(folders_list=folders_list, root_dir=root_dir, labels=labels, plot_type='PCA')



# Plot Elon Musk real videos vs Elon Musk fake videos  (PCA)
folders_list=[
    ['real/ElonMusk'],
    ['fake/ElonMusk'],
]
labels = ['Real Elon Musk Videos', 'Fake Elon Musk Videos']
vd.plot_features(folders_list=folders_list, root_dir=root_dir, labels=labels, plot_type='PCA')
vd.plot_features(folders_list=folders_list, root_dir=root_dir, labels=labels, plot_type='LDA')


# Plot Elon real/fake videos vs Obama's (LDA)
folders_list=[
    ['real/ElonMusk'],
    ['fake/ElonMusk'],
    ['real/Obama'],
    ['fake/Obama'],
]
labels = ['Real Elon Musk Videos', 'Fake Elon Musk Videos', 'Real Obama Videos', 'Fake Obama Videos']
vd.plot_features(folders_list=folders_list, root_dir=root_dir, labels=labels, plot_type='LDA')



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
vd.plot_features(folders_list=folders_list, root_dir=root_dir)
vd.plot_features(folders_list=folders_list, root_dir=root_dir, plot_type='LDA')


# real fake features in 2D for many subjects
folders_list=[
    ['real/Mattarella', 'real/ElonMusk', 'real/Obama', 'real/Renzi', 'real/QueenElisabeth'],
    ['fake/Mattarella', 'fake/Renzi', 'fake/Obama', 'fake/ElonMusk', 'fake/QueenElisabeth'],
]
vd.plot_features(folders_list=folders_list, root_dir=root_dir)
vd.plot_features(folders_list=folders_list, root_dir=root_dir, plot_type='LDA')



# Plot challenge real videos vs challenge fake videos
folders_list=[
    ['real/__challenge/'+x for x in os.listdir(f'{root_dir}/real/__challenge')],
    ['fake/__challenge/'+x for x in os.listdir(f'{root_dir}/real/__challenge')],
]
labels = ['Real Challenge Videos', 'Fake Challenge Videos']
vd.plot_features(folders_list=folders_list, root_dir=root_dir, labels=labels, plot_type='LDA')
vd.plot_features(folders_list=folders_list, root_dir=root_dir, labels=labels, plot_type='PCA')


print("THE END!")