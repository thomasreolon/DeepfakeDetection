import os, pathlib
from videoanalizer import VideoAnalizer

os.chdir(pathlib.Path(__file__).parent.absolute())
vd = VideoAnalizer()


############ Plot videos in a graph
ROOT_DIR = '../test_data/videos'
SAVE_PATH= '../output'
folders_list=[
    ['real/ElonMusk/train'],
    ['fake/ElonMusk'],
    ['real/Obama/train'],
    ['fake/Obama'],
    ['real/morez'],
    ['fake/morez'],
    ['real/Renzi'],
    ['fake/Renzi'],
    ['real/thomas1'],
    ['fake/thomas1'],
]
vd.plot_features(folders_list=folders_list, root_dir=ROOT_DIR, save_path=SAVE_PATH)
