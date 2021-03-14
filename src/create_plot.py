import os, pathlib
from videoanalizer import VideoAnalizer

os.chdir(pathlib.Path(__file__).parent.absolute())
vd = VideoAnalizer()


############ Plot videos in a graph
ROOT_DIR = '../test_data/videos'
SAVE_PATH= '../output'              # where plots are saved
folders_list=[
    # each sublist will have a different color in the plot
    ['real/ElonMusk/train'],        # relative path from ROOT_DIR
    ['fake/ElonMusk'],
    ['real/Obama/train'],
    ['fake/Obama'],
    ['real/morez'],                 
    ['fake/morez'],
]
vd.plot_features(folders_list=folders_list, root_dir=ROOT_DIR, save_path=SAVE_PATH, plot_type='LDA')
