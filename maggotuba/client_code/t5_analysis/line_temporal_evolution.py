import os
import itertools
import logging
import multiprocessing as mp

import numpy as np
from maggotuba.mmd import compute_linear_estimator_of_mmd
from scipy.spatial import distance_matrix
from tqdm import tqdm

import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt
import sklearn.cluster as skcl
from scipy.cluster.hierarchy import dendrogram
from sklearn.manifold import MDS
from sklearn.decomposition import PCA
from scipy.linalg import eigh
from scipy.spatial import distance_matrix

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

def iterlines(point_dynamics_root):
    for line in os.listdir(point_dynamics_root):
        if not(os.path.isdir(os.path.join(point_dynamics_root, line))
           and sum([line.endswith(s) for s in ['UAS_TNT_3_0014']])):#, 'UAS_TNT_2_0003', 'UAS_impTNT_2_0076']])):
            continue
        for protocol in os.listdir(os.path.join(point_dynamics_root, line)):
            if (not(os.path.isdir(os.path.join(point_dynamics_root, line, protocol)))
                 or not(protocol.startswith('p_8_45s1x30s0s#p_8_105s10x2s10s#n#n@100')
                     or protocol.startswith('p_3gradient1_45s1x30s0s#p_3gradient1_105s10x2s10s#n#n@100'))):
                continue
            yield line, protocol

def load_embedding(embeddings_dest_folder, line_folder, protocol_folder, start=0, duration=20, cutoff=None):
    filename = os.path.join(embeddings_dest_folder,
                            line_folder,
                            protocol_folder,
                            'encoded_trajs.npy')
    data = np.load(filename)
    if data.shape[1] != 150:
        print(line, folder, data.shape)
    data = data[:,start:start+duration,:]
    if cutoff:
        data = data[:cutoff//duration,:,:]
    return data.reshape(-1, data.shape[-1])


def compute_mmd_between_line(embeddings_folder, line1, protocol1, start1, line2, protocol2, start2, sigma, mmd_dest_file, duration=20):
        embedding1 = load_embedding(embeddings_folder, line1, protocol1, start=start1, duration=duration)
        embedding2 = load_embedding(embeddings_folder, line2, protocol2, start=start2, duration=duration)

        mmd, var, m = compute_linear_estimator_of_mmd(embedding1, embedding2, sigma)
        with open(mmd_dest_file, 'a') as f:
            f.write("{}, {}, {}, {}, {}, {}, {}, {}, {}\n".format(line1, protocol1, start1,
                                                                  line2, protocol2, start2,
                                                                  mmd, var, m))

def aux_compute_mmd_between_line(embeddings_folder, pair_of_lines, sigma, mmd_dest_file, duration):
    line1, line2 = pair_of_lines
    line1, start1 = line1
    line1, protocol1 = line1
    line2, start2 = line2
    line2, protocol2 = line2
    compute_mmd_between_line(embeddings_folder, line1, protocol1, start1, line2, protocol2, start2, sigma, mmd_dest_file, duration=duration)

def construct_mapping(mmd_file, effector=None):
    mapping = {}

    with open(mmd_file, 'r') as f:
        for line in f.readlines():
            line1, protocol1, start1, line2, protocol2, start2, _, _, _ = [s.strip(' \n') for s in line.split(',')]
            if effector is not None:
                if not(line1.endswith(effector) and line2.endswith(effector)):
                    continue
            if not((line1, protocol1, start1) in mapping):
                mapping[(line1, protocol1, start1)] = len(mapping)
            if not((line2, protocol2, start2) in mapping):
                mapping[(line2, protocol2, start2)] = len(mapping)
    return mapping

def construct_matrix(mmd_file, mapping, effector=None):
    matrix = np.zeros((len(mapping), len(mapping)))
    with open(mmd_file, 'r') as f:
        for line in f.readlines():
            line1, protocol1, start1, line2, protocol2, start2, mmd, _, _ = [s.strip(' \n') for s in line.split(',')]
            if effector is not None:
                if not(line1.endswith(effector) and line2.endswith(effector)):
                    continue
            mmd = float(mmd)
            i, j = mapping[(line1, protocol1, start1)], mapping[(line2, protocol2, start2)]
            matrix[i,j] = mmd
            matrix[j,i] = mmd
    return matrix

def main(embeddings_folder, duration, mmd_dest_file, n_procs):
    sigma = 5.5


    if not(os.path.isfile(mmd_dest_file)):
        # compute the mmd
        pool = mp.Pool(n_procs)
        iter_starts = [0,10,20,30,40]
        iter_lines_starts = itertools.product(iterlines(embeddings_folder), iter_starts)
        iter_pair_of_lines = itertools.permutations(iter_lines_starts, r=2)

        iter_args = zip(itertools.repeat(embeddings_folder),
                        iter_pair_of_lines,
                        itertools.repeat(sigma),
                        itertools.repeat(mmd_dest_file),
                        itertools.repeat(duration))

        pool.starmap(aux_compute_mmd_between_line, iter_args, chunksize=50)
        pool.close()
        pool.join()
        pool.terminate()

    # load matrix
    mapping = construct_mapping(mmd_dest_file)
    matrix = construct_matrix(mmd_dest_file, mapping)
    # plt.matshow(matrix)
    # plt.show()

    # Check spectrum
    delta = np.sqrt(matrix)
    H = np.eye(len(delta))-1/len(delta)*np.ones_like(delta)
    S = -0.5*H.dot(delta.dot(H))
    spec = eigh(S, eigvals_only=True)
    spec = np.real(spec)
    plt.figure()
    plt.hist(spec, bins=100)
    plt.show()
    plt.figure()
    plt.plot(np.sort(spec)[::-1])
    plt.show()

    # MDS
    mds = MDS(n_components=300, dissimilarity='precomputed')
    embedded_X = mds.fit_transform(delta)

    # 2D plot
    # UMAP = ParametricUMAP(n_components=2)
    # pca = PCA(n_components=2)
    # X_in_2d = pca.fit_transform(embedded_X)

    # data = pd.DataFrame(X_in_2d, columns=['x', 'y'])
    # protocols = []
    # lines_prots = []
    # lines = []
    # starts = []
    # for tup, val in mapping.items():
    #     line, protocol, start = tup
    #     lines.append(line)
    #     protocols.append(protocol)
    #     lines_prots.append(line+protocol)
    #     starts.append(start)
    # data['line'] = lines
    # data['protocol'] = protocols
    # data['effector'] = ['@'.join(l.split('@')[1:]) for l in lines]
    # data['start'] = starts
    # data['line_prot'] = lines_prots
    # fig = px.scatter(data, x='x', y='y', color='start', hover_data=['protocol', 'line', 'start'])
    # fig.show()


    # hierarchical clustering
    hierarch = skcl.AgglomerativeClustering(n_clusters=None, distance_threshold=100, compute_full_tree=True, linkage='complete', compute_distances=True)
    hierarch.fit(embedded_X)
    labels = hierarch.labels_
    order = create_column_order(hierarch)

    # Reordering of the matrix based on the hierarchical clustering
    perm = np.array(order)
    matrix = matrix[perm,:][:, perm]

    # # plot the dendrogram
    # plt.figure()
    # plt.title("Hierarchical Clustering Dendrogram")
    # plot_dendrogram(hierarch)
    # plt.xlabel("Number of points in node (or index of point if no parenthesis).")
    # plt.show()

    # Plot the reordered matrix
    plt.figure()
    plt.imshow(matrix)
    plt.colorbar()
    plt.show()

    

if __name__ == '__main__':
    embeddings_folder = '/home/alexandre/workspace/t5_embeddings'
    duration = 10 # corresponds to roughly 1 seconds of post-activation activity
    mmd_dest_file = '/home/alexandre/workspace/t5_embeddings/mmd{}_sliding1swindow.csv'.format(duration)

    n_procs = 10

    main(embeddings_folder, duration, mmd_dest_file, n_procs)