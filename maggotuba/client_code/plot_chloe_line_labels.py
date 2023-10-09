import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import Colormap
import pandas as pd

from sklearn.manifold import MDS
from umap import UMAP





def main(model, filename, tracker):

    # read the cluster file
    df = pd.read_csv(filename)

    clusters = list(df['lines'])
    cluster_names = df['0']
    for idx, cluster in enumerate(clusters):
        cluster = cluster[1:-1].split(',')
        cluster = [l.strip()[1:-1] for l in cluster]
        clusters[idx] = cluster

    # constuct label, a dict mapping lines to cluster labels
    label = {}
    for idx, cluster in enumerate(clusters):
        print(idx)
        print(cluster)
        print(70*'-')
        for line in cluster:
            label[line] = idx

    plt.figure()
    index, counts = np.unique([v for v in label.values()], return_counts=True)
    sorter = np.argsort(index)
    index = index[sorter]
    counts = counts[sorter]
    print(index, counts)
    cmap = plt.get_cmap('tab20')
    plt.pie(counts, labels=[f'clus_{c}' for c in index], colors=[cmap(i) for i in index])
    plt.show()
    
    assert(label[clusters[0][1]] == 0)
    assert(label[clusters[5][0]] == 5)

    # open mmd files
    mmd_arrays = np.load(os.path.join(model, 'training_log', 'experiment_1', 'mmd', f'{tracker}_mmd.npz'), allow_pickle=True)
    lines = mmd_arrays['lines_index']
    protocols = mmd_arrays['protocols_index']
    mmd = mmd_arrays['mmd2_quad']
    
    if tracker == 't5':
        # reformat lines
        # lines = np.array([l.split('@')[0] for l in lines])
        for line, protocol in zip(lines, protocols):
            assert(line+'/'+protocol in label)
        for l in label.keys():
            l, p = l.split('/')
            if not(p in protocols[lines==l]):
                print(l, p)

        # construct labels
        labels = np.array([label[line+'/'+protocol] for line, protocol in zip(lines, protocols)], dtype=int)

        # high dimensional MDS
        print('MDS...')
        mds = MDS(100, dissimilarity='precomputed')
        matrix = np.sqrt(np.maximum(mmd, 0.))
        X = mds.fit_transform(matrix)

        # UMAP to 2D
        print('UMAP...')
        umap = UMAP(n_neighbors=50)
        if bool('supervised') == True:
            X = umap.fit_transform(X, y=labels)
        if bool('unsupervised') == False:
            X = umap.fit_transform(X)

        # Scatter plot
        plt.figure()
        cmap = plt.get_cmap('tab20')
        plt.scatter(X[:,0], X[:,1], c=[cmap(l) for l in labels])
        plt.legend(handles=[Patch(fc=cmap(idx), label=l) for idx, l in enumerate(cluster_names)])
        plt.show()
    if tracker == 't15':
        protocol = 'r_LED100_30s2x15s30s#n#n#n@100'
        effector = 'UAS_Chrimson_Venus_X_0070'

        # remove wrong protocols
        mmd = mmd[protocols==protocol,:][:,protocol==protocols]
        lines = lines[protocols==protocol]
        protocols = protocols[protocols==protocol]

        # remove wrong effectors
        effector_indicator = np.array([l.endswith(effector) for l in lines])
        mmd = mmd[effector_indicator,:][:,effector_indicator]
        lines = lines[effector_indicator]
        protocols = protocols[effector_indicator]

        # reformat lines
        lines = np.array([l.split('@')[0] for l in lines])
        for line in lines:
            assert(line in label)

        # construct labels
        labels = np.array([label[line] for line in lines], dtype=int)

        # high dimensional MDS
        print('MDS...')
        mds = MDS(100, dissimilarity='precomputed')
        matrix = np.sqrt(np.maximum(mmd, 0.))
        X = mds.fit_transform(matrix)

        # UMAP to 2D
        print('UMAP...')
        umap = UMAP(n_neighbors=50)
        if bool('supervised') == True:
            X = umap.fit_transform(X, y=labels)
        if bool('unsupervised') == False:
            X = umap.fit_transform(X)

        # Scatter plot
        plt.figure()
        cmap = plt.get_cmap('tab20')
        plt.scatter(X[:,0], X[:,1], c=[cmap(l) for l in labels])
        plt.legend(handles=[Patch(fc=cmap(idx), label=l) for idx, l in enumerate(cluster_names)])
        plt.show()


if __name__ == '__main__':
    tracker = 't5'
    filename = os.path.join('/home/alexandre/Desktop', f'suffix_tree_clusters_{tracker}_lineprotocol_hierarch.csv')
    model = os.path.join('/home/alexandre/workspace', 'maggotuba_scale_40')
    main(model, filename, tracker)