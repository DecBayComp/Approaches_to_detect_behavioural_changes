# standard library packages
import os
import pickle
import json
import itertools
from multiprocessing import Pool

# utilities
from tqdm import trange

# scientific computing packages
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
from matplotlib.lines import Line2D
from matplotlib.colors import to_rgba
import scipy

import pandas as pd

# data science packages
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial import distance_matrix
from umap import UMAP
from umap.plot import connectivity

# homegrown packages
from maggotuba.mmd import compute_linear_estimator_of_mmd, pval
from maggotuba.mmd import cmmd2

# copy pasted from behavior_model/visualization/visualize.py

def alpha_shape(points, alpha, only_outer=True):
    """
    Compute the alpha shape (concave hull) of a set of points.
    https://stackoverflow.com/questions/23073170/calculate-bounding-polygon-of-alpha-shape-from-the-delaunay-triangulation
    :param points: np.array of shape (n,2) points.
    :param alpha: alpha value.
    :param only_outer: boolean value to specify if we keep only the outer border
    or also inner edges.
    :return: set of (i,j) pairs representing edges of the alpha-shape. (i,j) are
    the indices in the points array.
    """
    assert points.shape[0] > 3, "Need at least four points"

    def add_edge(edges, i, j):
        """
        Add a line between the i-th and j-th points,
        if not in the list already
        """
        if (i, j) in edges or (j, i) in edges:
            # already added
            assert (j, i) in edges, "Can't go twice over same directed edge right?"
            if only_outer:
                # if both neighboring triangles are in shape, it is not a boundary edge
                edges.remove((j, i))
            return
        edges.add((i, j))

    tri = scipy.spatial.Delaunay(points)
    edges = set()
    # Loop over triangles:
    # ia, ib, ic = indices of corner points of the triangle
    for ia, ib, ic in tri.simplices:
        pa = points[ia]
        pb = points[ib]
        pc = points[ic]
        # Computing radius of triangle circumcircle
        # www.mathalino.com/reviewer/derivation-of-formulas/derivation-of-formula-for-radius-of-circumcircle
        a = np.sqrt((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2)
        b = np.sqrt((pb[0] - pc[0]) ** 2 + (pb[1] - pc[1]) ** 2)
        c = np.sqrt((pc[0] - pa[0]) ** 2 + (pc[1] - pa[1]) ** 2)
        s = (a + b + c) / 2.0
        area = np.sqrt(s * (s - a) * (s - b) * (s - c))
        circum_r = a * b * c / (4.0 * area)
        if circum_r < alpha:
            add_edge(edges, ia, ib)
            add_edge(edges, ib, ic)
            add_edge(edges, ic, ia)
    return edges

def fatten(xy, eps=0.1):
    incr = np.empty_like(xy)
    xy = np.vstack([xy[-1:], xy, xy[:1]])
    for i in range(1, len(xy)-1):
        u1 = xy[i+1]-xy[i]
        u1 /= np.linalg.norm(u1)
        u2 = xy[i-1]-xy[i]
        u2 /= np.linalg.norm(u2)
        u = u1+u2
        u *= eps/np.linalg.norm(u)
        sign = 2*(np.linalg.det(np.hstack([u.reshape(-1,1), u1.reshape(-1,1)])) > 0) - 1
        incr[i-1] = sign*u
    return xy[1:-1]+incr

def compute_connectivity(umapper, labels):
    coo_graph = umapper.graph_.tocoo()
    edge_df = pd.DataFrame(
        np.vstack([coo_graph.row, coo_graph.col, coo_graph.data]).T,
        columns=("source", "target", "weight"),
    )
    edge_df["source"] = edge_df.source.astype(np.int32)
    edge_df["source"] = labels[edge_df["source"].to_numpy(int)]
    edge_df["target"] = edge_df.target.astype(np.int32)
    
    n_labels = len(set(labels))
    points_from_labels = {i: np.argwhere(labels==i) for i in range(n_labels)}
    total_weight = {}

    for i in range(n_labels):
        for j in range(i+1, n_labels):
            total_weight[i,j]  = np.sum(edge_df.loc[(edge_df["source"] == i) & (edge_df["target"] == j)]["weight"])
            total_weight[i,j] += np.sum(edge_df.loc[(edge_df["source"] == j) & (edge_df["target"] == i)]["weight"])

    return total_weight

def create_histogram(x, y, w, h, weights, p=0.9):
    behavior_colors = ['#17202A', '#C0392B', '#8BC34A', '#2E86C1', '#26C6DA', '#F1C40F', 'orange', 'magenta']
    n = len(weights)
    wbar = p*w/n
    wspace = (1-p)*w/(n-1)
    rectangles = []
    for i, weight in enumerate(weights):
        rectangles.append(Rectangle((x+i*(wbar+wspace),y), wbar, weight*h, fc=behavior_colors[i]))

    lines = []
    for p in [-0.05, -0.025, 0.025, 0.05]:
        lines.append(Line2D([x, x+w], [y+p*h, y+p*h], color='xkcd:dark red', ls='--', lw=1))
    lines.append(Line2D([x,x+w], [y,y], color='k', zorder=-100, lw=1))

    return rectangles, lines
# copy pasted from cli.cli_mmd and then modified

def add_arguments_plot_matrix(parser):
    parser.add_argument('--name', default=None)
    parser.add_argument('--tracker', default=None)
    parser.add_argument('--type', default='pval')
    parser.add_argument('--threshold', '-th', default=None, type=float)
    parser.add_argument('--rectify', default=False, action='store_true')

def plot_matrix(args, column_order=None):
    if args.tracker is None:
        raise ValueError('Please indicate a tracker.')
    if args.name is None:
        raise ValueError('Please indicate an experiment name.')


    directory = os.path.dirname(os.path.abspath(__file__))

    dir = os.getcwd()
    os.chdir(f'/home/alexandre/workspace/maggotuba_models/maggotuba_scale_{args.scale}')

    # get training log
    with open('config.json', 'r') as f:
        config = json.load(f)
        log_dir = config['log_dir']

    if args.type == 'dist':
        mmd_data = np.load(os.path.join(log_dir, args.name, 'mmd', f'{args.tracker}_mmd.npz'))
        mmd2 = mmd_data['mmd2_quad']
        
        dist = np.sqrt(np.maximum(mmd2,0))
        hierarch = AgglomerativeClustering(n_clusters=len(dist), compute_full_tree=True, linkage='complete', compute_distances=True, affinity='precomputed')
        hierarch.fit(dist)
        labels = hierarch.labels_
        if column_order is None:
            print('New column order')
            order = create_column_order(hierarch)
            column_order = order
        else:
            print('Reusing column order')
            order = column_order
        mmd2 = mmd2[:,order][order,:]
        
        plt.figure()
        if args.rectify:
            plt.imshow(np.sqrt(np.maximum(0,mmd2)))
        else:
            plt.imshow(mmd2)
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(os.path.join(directory, f'figure_4C_dist_{args.scale}.jpeg'))

    elif args.type == 'lines2D':
        args.scale = 20
        mmd_data = np.load(os.path.join(log_dir, args.name, 'mmd', f'{args.tracker}_mmd.npz'), allow_pickle=True)
        mmd2 = mmd_data['mmd2_quad']
        lines = mmd_data['lines_index']
        protocols = mmd_data['protocols_index']

        with open(os.path.join(directory, 'histogram_t5_20.pkl'), 'rb') as f:
            histograms = pickle.load(f)
        for l in histograms:
            for p in histograms[l]:
                histograms[l][p] = list(np.array(histograms[l][p][:-1])/np.sum(histograms[l][p][:-1]))


        if os.path.exists(os.path.join(directory, 'umap_lines.pkl')):
            with open(os.path.join(directory, 'umap_lines.pkl'), 'rb') as f:
                d = pickle.load(f)
            mds = d['mds']
            labels = d['labels']
            umap = d['umap']
            umapper = d['umapper']

        else:
            # MDSing the data
            print('mds...')
            centering = np.eye(mmd2.shape[0])-np.ones_like(mmd2)/mmd2.shape[0]
            mmd2_c = -0.5*centering.dot(mmd2).dot(centering)
            eigs, eigvs = np.linalg.eigh(mmd2_c)
            eigs, eigvs = eigs[eigs>0], eigvs[:,eigs>0]
            mds = eigvs.dot(np.sqrt(np.diag(eigs)))
            # hierarchical clustering
            print("agglom...")
            agglom = AgglomerativeClustering(n_clusters=5, linkage='ward')
            labels = agglom.fit_predict(mds)

            print('umap...')
            umapper = UMAP(min_dist=0.5, spread=0.5)
            umap =umapper.fit_transform(mds, y=labels)
            umap = umap - np.mean(umap, axis=0, keepdims=True)

            d = {'mds':mds, 'umap':umap, 'labels':labels, 'umapper':umapper}
            with open(os.path.join(directory, 'umap_lines.pkl'), 'wb') as f:
                pickle.dump(d, f)

        pd.DataFrame(dict(line=lines, protocol=protocols, cluster=labels)).to_csv(os.path.join(directory, f'cluster_labels.csv'))
        exit()
        plt.figure()

        cmap = plt.get_cmap('tab10')

        for idx in range(5):
            color = to_rgba('grey')
            # alpha blending
            dim_color = np.array(color)
            dim_color = (0.4*dim_color + 0.6*np.ones_like(dim_color))
            dim_color[3] = 1 
            dim_color = tuple(dim_color)

            cluster = umap[labels==idx]
            edges = list(alpha_shape(cluster, 0.4))
            ordered_edges = [edges.pop()]
            while edges:
                start = ordered_edges[-1][1]
                next_edge_idx = [i for i, e in enumerate(edges) if e[0] == start][0]
                ordered_edges.append(edges.pop(next_edge_idx))
            
            xy = np.empty((len(ordered_edges),2))
            for i, (e1, e2) in enumerate(ordered_edges):
                xy[i,:] = cluster[e1]

            xy = fatten(xy)
            poly = Polygon(xy, fc=dim_color, ec=color)
            plt.gca().add_patch(poly)
        
        means = [np.mean(umap[labels==i], axis=0) for i in range(5)]

        weights = compute_connectivity(umapper, labels)
        weights = {k:v for k, v in weights.items() if v > 0}
        s = sum(weights.values())
        m = min(weights.values())
        weights = {k:np.log(10*v/m)/np.log(3) for k, v in weights.items()}
    
        for i, j in weights:
                # xs = np.array()
                plt.plot([means[i][0], means[j][0]], [means[i][1], means[j][1]], lw=weights[i,j], color='grey', ls='--', zorder=-5)

        plt.scatter(umap[:,0], umap[:,1], s=4., marker='.', c='grey')


        mean_histogram = np.mean(np.array([histograms[l][p] for l, p in zip(lines, protocols)]), axis=0)
        dx = [  1.5, -6,  2.5, 2, -5.0]
        dy = [ -1.0, -1, -0.5, 0, -0.3]
        for i in range(5):
            lines_cl = lines[labels==i]
            protocols_cl = protocols[labels==i]
            hists_cl = [histograms[l][p] for l, p in zip(lines_cl, protocols_cl)]
            hists_cl = np.array(hists_cl)
            histo = np.mean(hists_cl, axis=0)
            histo = (histo-mean_histogram)
            rects, lines2d = create_histogram(means[i][0]+dx[i], means[i][1]+dy[i], 3, 30, histo)
            for rect in rects:
                plt.gca().add_patch(rect)
            for line in lines2d:
                plt.gca().add_artist(line)

        plt.axis('equal')
        plt.show()
        # plt.savefig(os.path.join(directory, f'figure_4D.eps'))

    os.chdir(dir)

    return column_order


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


if __name__ == '__main__':
    
    class args:
        pass
    args.name = 'experiment_1'
    args.tracker = 't5'
    args.type = 'dist'
    args.rectify = True

    # for scale in [20, 40]:
    #     args.scale = scale
    #     plot_matrix(args)

    args.scale = 20
    args.type = 'lines2D'
    plot_matrix(args)