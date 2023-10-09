import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import sklearn.cluster as skcl
from scipy.cluster.hierarchy import dendrogram
from sklearn.manifold import MDS
from scipy.linalg import eigh
from scipy.spatial import distance_matrix
import os
from MDSProb import MDS_prob
from tqdm import trange



from umap.parametric_umap import ParametricUMAP
import plotly.express as px
import pandas as pd
from maggotuba.mmd import pval


def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)

def create_column_order(model):
    N = model.n_leaves_
    order = [2*N-2]
    while not(np.all(np.array(order)<N)):
        new_order = []
        for node in order:
            if node >= N:
                new_order.append(model.children_[node-N][0])
                new_order.append(model.children_[node-N][1])
            else:
                new_order.append(node)
        order = new_order
    return new_order



def construct_mapping(mmd_file, effector=None):
    mapping = {}

    with open(mmd_file, 'r') as f:
        for line in f.readlines():
            line1, protocol1, line2, protocol2, _, _, _ = [s.strip(' \n') for s in line.split(',')]
            if effector is not None:
                if not(line1.endswith(effector) and line2.endswith(effector)):
                    continue
            if not((line1, protocol1) in mapping):
                mapping[(line1, protocol1)] = len(mapping)
            if not((line2, protocol2) in mapping):
                mapping[(line2, protocol2)] = len(mapping)
    return mapping

def construct_matrix(mmd_file, mapping, effector=None):
    matrix = np.zeros((len(mapping), len(mapping)))
    pval_matrix = np.ones((len(mapping), len(mapping)))
    var_matrix = np.ones((len(mapping), len(mapping)))
    with open(mmd_file, 'r') as f:
        for line in f.readlines():
            line1, protocol1, line2, protocol2, mmd, sig2, m = [s.strip(' \n') for s in line.split(',')]
            if effector is not None:
                if not(line1.endswith(effector) and line2.endswith(effector)):
                    continue
            mmd = float(mmd)
            m = int(m)
            sig2 = float(sig2)
            i, j = mapping[(line1, protocol1)], mapping[(line2, protocol2)]
            matrix[i,j] = mmd
            matrix[j,i] = mmd
            if sig2 > 0:
                pval_matrix[i,j] = pval(mmd, sig2, m) 
                pval_matrix[j,i] = pval_matrix[i,j]
            var_matrix[i,j] = 2./m*sig2
            var_matrix[j,i] = var_matrix[i,j]
    print(np.isnan(matrix).any())
    return matrix, pval_matrix, var_matrix

def load_histograms(histogram_dir, mapping):
    histo = np.zeros((len(mapping), 6))
    for line, protocol in mapping.keys():
        histo_path = os.path.join(histogram_dir, line, protocol, 'histogram_10.npy')
        histo[mapping[(line, protocol)]] = np.load(histo_path)
    return histo

def plot_scatter_pie(X, histo_arr, diff = False):
    def plot_bar_chart(x, y, bar_width, bar_height, histo):
        corner_x = np.arange(x-3*bar_width, x+3*bar_width, bar_width)
        corner_y = np.repeat(y,6)
        rectangles = [patches.Rectangle((cx, cy), 0.9*bar_width, bar_height*h, fc=c) for cx, cy, h, c in zip(corner_x,
                                                                                                       corner_y,
                                                                                                       histo,
                                                                                                       ['black', 'red', 'green', 'mediumblue', 'lightblue', 'yellow'])]
        return rectangles

    ncl=12
    kmeans = skcl.KMeans(ncl)
    labels = kmeans.fit_predict(X)

    centers = kmeans.cluster_centers_
    mean_histo = np.mean(histo_arr, axis=0)
    aggregated_histos = np.zeros((ncl, 6))
    for cl in range(ncl):
        aggregated_histos[cl,:] = np.mean(histo_arr[labels==cl,:], axis=0)


    bar_width = 0.15
    plt.figure()
    for xy, histo in zip(centers, aggregated_histos):
        if diff:
            rescale=False
            if rescale:
                bar_height = 1.
                rectangles = plot_bar_chart(xy[0], xy[1], bar_width, bar_height, (histo-mean_histo)/mean_histo)
            else:
                bar_height = 10.
                rectangles = plot_bar_chart(xy[0], xy[1], bar_width, bar_height, (histo-mean_histo))
        else:
            bar_height = 1.5
            rectangles = plot_bar_chart(xy[0], xy[1], bar_width, bar_height, histo)
        for r in rectangles:
            plt.gca().add_artist(r)
    plt.scatter(X[:,0], X[:,1], alpha=0.)
    plt.show()

def plot_histo_grid(X, histo_arr):
    fig, axs = plt.subplots(2,3)
    labels = ['run', 'bend', 'stop', 'hunch', 'back', 'roll']
    for i, ax in enumerate(axs.flatten()):
        sp = ax.scatter(X[:,0], X[:,1], c=histo_arr[:,i])
        ax.set_title(f'% {labels[i]}')
        fig.colorbar(sp, ax=ax)
    plt.show()


def main(mmd_file, histogram_dir, effector=None):
    print('Loading mapping...')
    mapping = construct_mapping(mmd_file, effector=effector)
    print('Loading matrix...')
    matrix, pval_matrix, var_matrix = construct_matrix(mmd_file, mapping, effector=effector)
    print('Loading histograms...')
    histo_arr = load_histograms(histogram_dir, mapping)

    rectified_matrix = np.sqrt(np.maximum(matrix, np.zeros_like(matrix)))

    # kmeans = skcl.KMeans(n_clusters=20)
    # labels = kmeans.fit_predict(rectified_matrix)
    # perm = np.concatenate([np.argwhere(labels==l) for l in range(8)]).flatten()
    # rectified_matrix = rectified_matrix[perm,:][:, perm]
    
    # plt.figure()
    # plt.title("Hierarchical Clustering Dendrogram")
    # plot_dendrogram(agglom, truncate_mode="level", p=20)
    # plt.xlabel("Number of points in node (or index of point if no parenthesis).")
    # plt.show()
    
    # plt.figure()
    # plt.imshow(rectified_matrix)
    # plt.show()
    # H = np.eye(len(matrix))-1/len(matrix)*np.ones_like(matrix)
    # S = -0.5*H.dot(matrix.dot(H))
    # spec = eigh(S, eigvals_only=True)
    # spec = np.real(spec)
    # plt.figure()
    # plt.hist(spec, bins=100)
    # plt.show()
    # plt.figure()
    # plt.plot(np.sort(spec)[::-1])
    # plt.show()

    # print("Performing MDS in 100D...")
    # mds = MDS(n_components=100, dissimilarity='precomputed')
    # embedded_X = mds.fit_transform(rectified_matrix)

    # print("Performing UMAP in 2D...")
    # UMAP = ParametricUMAP(n_components=2)
    # X_in_2d = UMAP.fit_transform(embedded_X)

    # print('Performing probabilistic MDS in 20D')
    # mds = MDS_prob(target_dim=100)
    # embedded_X = mds.fit_transform(matrix, var_matrix, n_steps=200)

    print('Assessing intrisic dimensionality using probabilistic MDS')
    logprobs = []
    for d in trange(1,50):
        mds = MDS_prob(target_dim=d)
        mds.fit(matrix, var_matrix, n_steps=200)
        logprobs.append(mds.loss)

    fig, axs = plt.subplots(1,2, sharex=True)
    axs[0].plot(logprobs)
    axs[1].semilogy(logprobs)
    fig.suptitle('Reconstruction error in function of MDS dimension')
    plt.show()



    # print("Performing UMAP in 2D...")
    UMAP = ParametricUMAP(n_components=2)
    X_in_2d = UMAP.fit_transform(embedded_X)

    # print('Plotting the bar charts...')
    # plot_scatter_pie(X_in_2d, histo_arr)
    print('Plotting the bar charts...')
    plot_scatter_pie(X_in_2d, histo_arr, diff=True)
    print("Plotting the grid of histograms...")
    plot_histo_grid(X_in_2d, histo_arr)

    # mds_prob = MDS_prob()
    # X_in_2d = mds_prob.fit_transform(matrix, np.sqrt(var_matrix))

    # data = pd.DataFrame(X_in_2d, columns=['x', 'y'])
    # protocols = []
    # lines = []
    # for tup, val in mapping.items():
    #     line, protocol = tup
    #     lines.append(line)
    #     protocols.append(protocol)
    # data['line'] = lines
    # data['protocol'] = protocols
    # data['effector'] = ['@'.join(l.split('@')[1:]) for l in lines]
    # data = data.loc[data['effector'].isin(['UAS_TNT_2_0003', 'UAS_TNT_3_0014', 'UAS_impTNT_2_0076'])]
    # # print(data.head())
    # fig = px.scatter(data, x='x', y='y', color='line', hover_data=['protocol', 'line'])
    # fig.show() 


    # ncl = 150

    # # kmeans = skcl.KMeans(n_clusters=ncl)
    # # labels = kmeans.fit_predict(embedded_X)
    # # perm = np.concatenate([np.argwhere(labels==l) for l in range(ncl)]).flatten()
    # # matrix = matrix[perm,:][:, perm]

    # trim the nans
    # not_nans = [not(np.sum(np.isnan(row))>1000) for row in matrix] # quick and dirty
    # matrix = matrix[not_nans,:][:, not_nans]
    # pval_matrix = pval_matrix[not_nans,:][:, not_nans]

    # not_unpvaluable = [not(np.sum(row==1)>1000) for row in pval_matrix] # quick and dirty
    # matrix = matrix[not_unpvaluable,:][:, not_unpvaluable]
    # pval_matrix = pval_matrix[not_unpvaluable,:][:, not_unpvaluable]

    # ncl = 150

    # plt.matshow(pval_matrix)
    # plt.show()
    # matrix = 1-pval_matrix
    # hierarch = skcl.AgglomerativeClustering(n_clusters=ncl, compute_full_tree=True, linkage='complete', compute_distances=True, affinity='precomputed')
    # hierarch.fit(rectified_matrix)
    # labels = hierarch.labels_
    # order = create_column_order(hierarch)

    # perm = np.concatenate([np.argwhere(labels==l) for l in range(ncl)]).flatten()
    # perm = np.array(order)
    # matrix_perm = matrix[perm,:][:, perm]
    # pval_matrix_perm = pval_matrix[perm,:][:, perm]

    # plt.figure()
    # plt.imshow(matrix_perm)
    # plt.colorbar()
    # plt.show()

    # hierarch.n_clusters = ncl
    # perm = np.concatenate([np.argwhere(labels==l) for l in range(ncl)]).flatten()
    # pval_matrix_perm = pval_matrix[perm,:][:, perm]

    # plt.figure()
    # plt.title("Hierarchical Clustering Dendrogram")
    # plot_dendrogram(hierarch)
    # plt.xlabel("Number of points in node (or index of point if no parenthesis).")
    # plt.show()

    # plt.figure()
    # plt.imshow(pval_matrix_perm>0.5)
    # plt.colorbar()
    # plt.show()
if __name__ == '__main__':
    mmd_file = '/home/alexandre/workspace/t5_embeddings/mmdtraj20.csv'
    histogram_dir = '/home/alexandre/workspace/t5_embeddings/histograms'
    effector = 'UAS_TNT_2_0003'
    # effector = 'UAS_Shi_ts1_3_0001'
    # effector = 'DL_UAS_GAL80ts_Kir21_23_0010'
    # effector = 'UAS_dTrpA1_2_0012'
    # effector = 'UAS_GAL80ts_Kir21_23_0004'
    # effector = 'UAS_TNT_3_0014'
    # effector = None
    main(mmd_file, histogram_dir, effector=effector)