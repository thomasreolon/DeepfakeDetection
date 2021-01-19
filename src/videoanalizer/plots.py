import matplotlib.pyplot as plt
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
import random

colors = ['#ff5733', '#ff9333','#ffca33','#fff033','#b1cf29','#b24516','#b68e51','#8b2c00','#ff1212','#ff7e95','#ff7ec2','#fa2d65','#fa2da6','#a5013f',]
random.shuffle(colors)

def plot_features2D(data, labels, ptype='PCA'):
    """
    data is a list of (list of samples)
    ttype can be choosen between PCA and LDA 
    """
    X=[]
    y=[]
    for i,d in enumerate(data):
        X+=d
        y+=[i for _ in range(len(d))]

    # type of projection
    n_components=2
    if (ptype=='PCA'):
        clf = PCA(n_components=n_components)
    else:
        n_components = min(n_components, len(labels)-1)
        clf = LinearDiscriminantAnalysis(n_components=n_components)

    clf.fit(X, y)

    # plot results
    plt.title('PCA')
    for d,c,l,i in zip(data,colors,labels,list(range(20))):
        coords = np.array(clf.transform(d))
        if (n_components==2):
            plt.scatter(coords[:,0], coords[:,1], color=c, label=l, alpha=0.7)
        else:
            plt.scatter(coords[:,0], np.zeros_like(coords[:,0])+i, color=c, label=l, alpha=0.7)

    if (n_components==1):
        plt.ylim((-3,3+len(data)))
    plt.legend()
    plt.show()










