import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
import random
from .covariance import CORRELATION_LABELS
matplotlib.use('Agg')

colors = ['#ff5733', '#ff9333','#ffca33','#fff033','#b1cf29','#b24516','#b68e51','#8b2c00','#ff1212','#ff7e95','#ff7ec2','#fa2d65','#fa2da6','#a5013f',]
random.shuffle(colors)
 

# removes NaN and infinites from the data
def clean(data):
    X, Y = [], []
    for i,d in enumerate(data):
        for x in d:
            x = np.array(x, dtype='float64')
            if np.isfinite(x).all():
                X.append(x)
                Y.append(i)
    return np.array(X, dtype='float64'), Y


# saves the plot of the features
def plot_features2D(data,out_dir, labels, ptype='PCA'):
    """
    data is a list of (list of samples)
    type can be choosen between PCA and LDA 
    """
    
    # get X as a list of points and y as their label
    X,y = clean(data)

    # type of projection
    n_components=2
    if (ptype=='PCA'):
        clf = PCA(n_components=n_components)
    else:
        n_components = min(n_components, len(labels)-1)
        clf = LinearDiscriminantAnalysis(n_components=n_components)

    clf.fit(X, y)

    # plot results
    tit = f'{ptype} {len(labels)} {labels[0]}'.replace('/', '-')
    plt.title(tit)
    data_result = []
    for d,c,l,i in zip(data,colors,labels,list(range(20))):
        clean_d = []
        for x in d:
            if (np.isfinite(np.array(x,dtype='float64')).all()):
                clean_d.append(x)
        coords = np.array(clf.transform(clean_d))
        if (n_components==2):
            plt.scatter(coords[:,0], coords[:,1], color=c, label=l, alpha=0.7)
        else:
            plt.scatter(coords[:,0], np.zeros_like(coords[:,0])+i, color=c, label=l, alpha=0.7)
        data_result.append(coords)
    if (n_components==1):
        plt.ylim((-3,3+len(data)))
    plt.legend()
    plt.savefig(f'{out_dir}/{tit}.png')           # save plot
    plt.clf()
    np.save(f'{out_dir}/{tit}.array', data_result)# save low dimensional space


    if (ptype=='PCA'):
        plt.matshow(clf.components_,cmap='magma')
        plt.yticks([0,1],['1st Comp','2nd Comp'],fontsize=7)
        plt.colorbar()
        plt.xticks(range(len(CORRELATION_LABELS)),CORRELATION_LABELS,rotation=65,ha='left', fontsize=3)
        plt.tight_layout()
        plt.savefig(f'{out_dir}/{tit}-features.png', dpi=300)           # save linear features
        plt.clf()
