# standard library packages
import os
import json
import itertools
from multiprocessing import Pool

# utilities
from tqdm import trange

# scientific computing packages
import numpy as np
import matplotlib.pyplot as plt

# data science packages
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial import distance_matrix
from umap import UMAP

# homegrown packages
from maggotuba.mmd import compute_linear_estimator_of_mmd, pval
from maggotuba.mmd import cmmd2

####################################################
#
#              COMPUTE THE MMD MATRIX
#
#####################################################

def add_arguments_cmpmat(parser):
    parser.add_argument('--tracker', type=str, default=None)
    parser.add_argument('--sigma', default=None)
    parser.add_argument('--name', default=None)
    parser.add_argument('--n_workers', default=10, type=int)

def compute_matrix(args):
    if args.tracker is None:
        raise ValueError('Please indicate a tracker.')
    if args.name is None:
        raise ValueError('Please indicate an experiment name.')

    # get training log
    with open('config.json', 'r') as f:
        config = json.load(f)
        log_dir = config['log_dir']
    
    # load config
    config_path = os.path.join(log_dir, args.name, 'config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    embeddings_folder = os.path.join(config['log_dir'], 'embeddings', args.tracker)

    mmd_dest_file = os.path.join(config['log_dir'], 'mmd', f'{args.tracker}_mmd.csv')
    os.makedirs(os.path.join(config['log_dir'], 'mmd'), exist_ok=True)

    compute_matrix_main(embeddings_folder, mmd_dest_file, args.n_workers, args.sigma)

# helper functions and compute_matrix_main

def iterlines(point_dynamics_root):
    for line in os.listdir(point_dynamics_root):
        if not os.path.isdir(os.path.join(point_dynamics_root, line)):
            continue
        for protocol in os.listdir(os.path.join(point_dynamics_root, line)):
            if not(os.path.isdir(os.path.join(point_dynamics_root, line, protocol))):
                continue
            if (point_dynamics_root.endswith('t5') 
                and not(protocol.startswith('p_8_45s1x30s0s#p_8_105s10x2s10s#n#n@100')
                     or protocol.startswith('p_3gradient1_45s1x30s0s#p_3gradient1_105s10x2s10s#n#n@100'))):
                continue

            yield line, protocol

def load_embedding(embeddings_dest_folder, line_folder, protocol_folder, cutoff=None):
    filename = os.path.join(embeddings_dest_folder,
                            line_folder,
                            protocol_folder,
                            f'encoded_trajs.npy')

    data = np.load(filename)
    if cutoff:
        data = data[:cutoff]
    return data

def compute_deciles_one_line(embeddings_folder, line_folder, protocol_folder):
        embedding = load_embedding(embeddings_folder, line_folder, protocol_folder, cutoff=20000)
        distmat = distance_matrix(embedding, embedding)
        m = distmat.shape[0]
        deciles = np.quantile(distmat[np.triu_indices(m,1,m)], np.linspace(0,1,11))

        return deciles, len(embedding)

def aggregate_deciles(N, deciles):
    weights = np.array([0.05, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.05 ]).reshape(1,-1)
    N = N.reshape(-1, 1)
    weights = N*weights
    weights = weights.flatten()
    deciles = deciles.flatten()
    argsort = np.argsort(deciles)
    deciles = deciles[argsort]
    weights = weights[argsort]
    cumweights = np.cumsum(weights)
    median_index = np.searchsorted(cumweights, 0.5*np.sum(weights))-1
    median = deciles[median_index]

    return median

def compute_mmd_between_line(embeddings_folder, pair_of_lines, sigma, mmd_dest_file):
    line1, line2 = pair_of_lines
    line1, protocol1 = line1
    line2, protocol2 = line2
    try:
        embedding1 = load_embedding(embeddings_folder, line1, protocol1)
        embedding2 = load_embedding(embeddings_folder, line2, protocol2)
    except FileNotFoundError:
        return
    mmd2, var, m = compute_linear_estimator_of_mmd(embedding1, embedding2, sigma)
    cmmd2_ = cmmd2(embedding1, embedding2, sigma)

    with open(mmd_dest_file, 'a') as f:       
        f.write("{}, {}, {}, {}, {}, {}, {}, {}\n".format(line1, protocol1,
                                                line2, protocol2,
                                                mmd2, var, m, cmmd2_))

def construct_mapping(mmd_file, effector=None):
    mapping = {}

    with open(mmd_file, 'r') as f:
        for line in f.readlines():
            line1, protocol1, line2, protocol2, _, _, _, _ = [s.strip(' \n') for s in line.split(',')]
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
    c_matrix = np.zeros((len(mapping), len(mapping)))
    pval_matrix = np.ones((len(mapping), len(mapping)))
    var_matrix = np.ones((len(mapping), len(mapping)))

    with open(mmd_file, 'r') as f:
        for line in f.readlines():
            line1, protocol1, line2, protocol2, mmd2, sig2, m, cmmd2_ = [s.strip(' \n') for s in line.split(',')]
            if effector is not None:
                if not(line1.endswith(effector) and line2.endswith(effector)):
                    continue
            mmd2 = float(mmd2)
            m = int(m)
            sig2 = float(sig2)
            i, j = mapping[(line1, protocol1)], mapping[(line2, protocol2)]
            matrix[i,j] = mmd2
            matrix[j,i] = mmd2
            c_matrix[i,j] = cmmd2_
            c_matrix[j,i] = cmmd2_
            if sig2 > 0:
                pval_matrix[i,j] = pval(mmd2, sig2, m) 
                pval_matrix[j,i] = pval_matrix[i,j]
            var_matrix[i,j] = 2./m*sig2
            var_matrix[j,i] = var_matrix[i,j]

    return matrix, pval_matrix, var_matrix, c_matrix

def compute_matrix_main(embeddings_folder, mmd_dest_file, n_procs, sigma=None):
    if sigma is None:
        # compute the kernel size
        print('Computing the kernel size...')
        # compute deciles
        deciles = []
        N = []
        for line_folder, protocol_folder in iterlines(embeddings_folder):
            try:
                deciles_line, N_line = compute_deciles_one_line(embeddings_folder, line_folder, protocol_folder)
            except FileNotFoundError:
                continue
            deciles.append(deciles_line)
            N.append(N_line)

        # aggregate deciles to compute median
        deciles = np.vstack(deciles)
        N = np.array(N)
        sigma = aggregate_deciles(N, deciles)
    print('sigma : ', sigma)


    print("Computing mmd and writing to .csv file.")
    # compute the mmd
    pool = Pool(n_procs)
    iter_pair_of_lines = itertools.permutations(iterlines(embeddings_folder), r=2)
    iter_args = zip(itertools.repeat(embeddings_folder),
                    iter_pair_of_lines,
                    itertools.repeat(sigma),
                    itertools.repeat(mmd_dest_file))

    pool.starmap(compute_mmd_between_line, iter_args, chunksize=50)
    pool.close()
    pool.join()
    pool.terminate()

    print("Storing data in more user-friendly formats.")
    mapping = construct_mapping(mmd_dest_file)
    mmd_matrix, pval_matrix, var_matrix, c_matrix = construct_matrix(mmd_dest_file, mapping)
    lines = np.empty(len(mapping), dtype=object)
    protocols = np.empty(len(mapping), dtype=object)
    for k, v in mapping.items():
        lines[v] = k[0]
        protocols[v] = k[1]

    to_store = {}
    to_store['lines_index'] = lines
    to_store['protocols_index'] = protocols
    to_store['mmd2'] = mmd_matrix
    to_store['var_mmd2'] = var_matrix
    to_store['pval_mmd2'] = pval_matrix
    to_store['sigma'] = sigma
    to_store['mmd2_quad'] = c_matrix
    filename = os.path.splitext(mmd_dest_file)[0]+'.npz'
    np.savez(filename, **to_store)


############################################################
#
#  Plot the data from the mmd matrix in a variety of ways
#
############################################################


def add_arguments_plot_matrix(parser):
    parser.add_argument('--name', default=None)
    parser.add_argument('--tracker', default=None)
    parser.add_argument('--type', default='pval')
    parser.add_argument('--threshold', '-th', default=None, type=float)
    parser.add_argument('--rearrange', default=False, action='store_true')
    parser.add_argument('--rectify', default=False, action='store_true')

def plot_matrix(args):
    if args.tracker is None:
        raise ValueError('Please indicate a tracker.')
    if args.name is None:
        raise ValueError('Please indicate an experiment name.')

    # get training log
    with open('config.json', 'r') as f:
        config = json.load(f)
        log_dir = config['log_dir']

    if args.type == 'pval':
        mmd_data = np.load(os.path.join(log_dir, args.name, 'mmd', f'{args.tracker}_mmd.npz'))
        pvals = mmd_data['pval_mmd2']
        if False: # bonferonni
            pvals = np.minimum(n*(n-1)/2*pvals, 1)

        if args.rearrange:
            matrix = np.tan(2/np.pi*(1-pvals))
            hierarch = AgglomerativeClustering(n_clusters=len(matrix), compute_full_tree=True, linkage='complete', compute_distances=True, affinity='precomputed')
            hierarch.fit(matrix)
            labels = hierarch.labels_
            order = create_column_order(hierarch)
            pvals = pvals[order,:][:,order]

        if args.threshold is not None:
            n = len(pvals)
            pvals = pvals > (args.threshold)#/(n*(n-1)/2))

        plt.figure()
        plt.imshow(pvals)
        plt.colorbar()
        plt.show()

    elif args.type == 'dist':
        mmd_data = np.load(os.path.join(log_dir, args.name, 'mmd', f'{args.tracker}_mmd.npz'))
        mmd2 = mmd_data['mmd2_quad']
        
        dist = np.sqrt(np.maximum(mmd2,0))
        hierarch = AgglomerativeClustering(n_clusters=len(dist), compute_full_tree=True, linkage='complete', compute_distances=True, affinity='precomputed')
        hierarch.fit(dist)
        labels = hierarch.labels_
        order = create_column_order(hierarch)
        mmd2 = mmd2[:,order][order,:]
        
        if args.rectify:
            plt.figure()
            plt.imshow(np.sqrt(np.maximum(0,mmd2)))
            plt.colorbar()
            plt.title(r'Estimator of $MMD$ for tracker {}'.format(args.tracker))
            plt.show()
        else:
            plt.figure()
            plt.imshow(mmd2)
            plt.colorbar()
            plt.title(r'Estimator of $MMD^2$ for tracker {}'.format(args.tracker))
            plt.show()

    elif args.type == 'lines2D':
        mmd_data = np.load(os.path.join(log_dir, args.name, 'mmd', f'{args.tracker}_mmd.npz'))
        mmd2 = mmd_data['mmd2_quad']

        print('mds...')
        centering = np.eye(mmd2.shape[0])-np.ones_like(mmd2)/mmd2.shape[0]
        mmd2_c = -0.5*centering.dot(mmd2).dot(centering)
        eigs, eigvs = np.linalg.eigh(mmd2_c)
        eigs, eigvs = eigs[eigs>0], eigvs[:,eigs>0]
        mds = eigvs.dot(np.sqrt(np.diag(eigs)))

        # print('umap...')
        umap = UMAP().fit_transform(mds)

        plt.figure()
        plt.scatter(umap[:,0], umap[:,1])
        plt.show()

    else:
        raise ValueError('Wrong matrix type')

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

######################################################
#
#              FIND HITS BASED ON TWO SAMPLE TEST
#
##########################################################

def add_argument_find_hits(parser):
    parser.add_argument('--job_id', type=int, default=0)
    parser.add_argument('--total_jobs', type=int, default=1)
    parser.add_argument('--root', default='experiment_1')
    parser.add_argument('--tracker', '-t', default='t5')
    parser.add_argument('--protocol', '-p', default='p_8_45s1x30s0s#p_8_105s10x2s10s#n#n@100')
    parser.add_argument('--effector', '-e', default='UAS_TNT_2_0003')
    parser.add_argument('--prefix', '-px', default='GMR')
    parser.add_argument('--reference', '-ref', default='FCF_attP2_1500062')
    parser.add_argument('--nboot', type=int, default=1000)

def find_hits(args):
    # for all lines in tracker with the proper prefix and effector:
    #     for the given protocol:
    #         compute the MMD between the reference and the line
    #         compute a p value using either spectral chi squared or the actual bootstrap
    #         store the tuple p-value, line

    from maggotuba.mmd import cscalar_square, cscalar_product, cmmd2, cbootstrap
    import numpy as np
    import json

    # collect all lines
    collected_lines = []
    for line in os.listdir(os.path.join(args.root, 'embeddings', args.tracker)):
        if line.startswith(args.prefix):
            effector = line.split(('@'))[1]
            maybe_numbers = line.split('_')[1][0]
            if maybe_numbers.isnumeric() and args.effector == effector:
                if args.protocol in os.listdir(os.path.join(args.root, 'embeddings', args.tracker, line)):
                    collected_lines.append(line)

    # select lines to deal with
    collected_lines = [l for i, l in enumerate(collected_lines) if not((i-args.job_id)%args.total_jobs)]


    # load reference
    ref_path = os.path.join(args.root, 'embeddings', args.tracker, '@'.join([args.reference, args.effector]), args.protocol, 'encoded_trajs.npy')
    reference = np.load(ref_path)
    sigma = np.median(distance_matrix(reference, reference))


    #############################################################
    #
    #           TIME EFFICIENT CYTHON BOOTSTRAP !
    #
    ##########################################################

    # process lines and write results to bootstrap.csv
    pvals = []
    for line in collected_lines:
        line_path = os.path.join(args.root, 'embeddings', args.tracker, line, args.protocol, 'encoded_trajs.npy')
        data = np.load(line_path)
 
        testStat, bootstrap = cbootstrap(data, reference, sigma, args.nboot)
        bootstrap = np.sort(bootstrap)
        pos = np.searchsorted(bootstrap, testStat)
        with open(os.path.join(args.root, 'bootstrap.csv'), 'a') as f:
            f.write(f"{line}, {pos}, {nboot}\n")
