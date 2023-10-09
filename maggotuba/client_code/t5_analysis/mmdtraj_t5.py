import os
import itertools
import logging
import multiprocessing as mp

import numpy as np
from maggotuba.mmd import compute_linear_estimator_of_mmd
from scipy.spatial import distance_matrix
from tqdm import tqdm



def iterlines(point_dynamics_root):
    for line in os.listdir(point_dynamics_root):
        if not os.path.isdir(os.path.join(point_dynamics_root, line)):
            continue
        for protocol in os.listdir(os.path.join(point_dynamics_root, line)):
            if (not(os.path.isdir(os.path.join(point_dynamics_root, line, protocol)))
                 or not(protocol.startswith('p_8_45s1x30s0s#p_8_105s10x2s10s#n#n@100')
                     or protocol.startswith('p_3gradient1_45s1x30s0s#p_3gradient1_105s10x2s10s#n#n@100'))):
                continue
            yield line, protocol

def load_embedding(embeddings_dest_folder, line_folder, protocol_folder, duration=20, cutoff=None):
    filename = os.path.join(embeddings_dest_folder,
                            line_folder,
                            protocol_folder,
                            'encoded_trajs.npy')
    data = np.load(filename)
    data = data[:,:duration,:]
    if cutoff:
        data = data[:cutoff//duration,:,:]
    data = data.reshape(data.shape[0], -1)
    return data

def compute_deciles_one_line(embeddings_folder, line_folder, protocol_folder, duration=20):
        embedding = load_embedding(embeddings_folder, line_folder, protocol_folder, duration=duration, cutoff=20000)
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

def compute_mmd_between_line(embeddings_folder, line1, protocol1, line2, protocol2, sigma, mmd_dest_file, duration=20):
        embedding1 = load_embedding(embeddings_folder, line1, protocol1, duration=duration)
        embedding2 = load_embedding(embeddings_folder, line2, protocol2, duration=duration)

        if (len(embedding1) <= 1) or (len(embedding2) <= 1):
            mmd, var, m = np.nan, np.nan, np.nan
        else:
            mmd, var, m = compute_linear_estimator_of_mmd(embedding1, embedding2, sigma)

        with open(mmd_dest_file, 'a') as f:
            f.write("{}, {}, {}, {}, {}, {}, {}\n".format(line1, protocol1,
                                                    line2, protocol2,
                                                    mmd, var, m))

def aux_compute_mmd_between_line(embeddings_folder, pair_of_lines, sigma, mmd_dest_file, duration):
    line1, line2 = pair_of_lines
    line1, protocol1 = line1
    line2, protocol2 = line2
    compute_mmd_between_line(embeddings_folder, line1, protocol1, line2, protocol2, sigma, mmd_dest_file, duration=duration)

def main(embeddings_folder, duration, mmd_dest_file, n_procs):
    # compute the kernel size
    # compute deciles
    # deciles = []
    # N = []
    # with tqdm(total=31) as pbar:
    #     for line_folder, protocol_folder in iterlines(embeddings_folder):
    #         if line_folder.startswith('FCF_'):
    #             deciles_line, N_line = compute_deciles_one_line(embeddings_folder, line_folder, protocol_folder, duration=duration)
    #             deciles.append(deciles_line)
    #             N.append(N_line)
    #             pbar.update()

    # # aggregate deciles to compute median
    # deciles = np.vstack(deciles)
    # N = np.array(N)
    # sigma = aggregate_deciles(N, deciles)
    # print('sigma : ', sigma)
    # sigma = 5.5
    # sigma = 1.0
    sigma = 16.5


    # compute the mmd
    pool = mp.Pool(n_procs)
    iter_pair_of_lines = itertools.permutations(iterlines(embeddings_folder), r=2)
    iter_args = zip(itertools.repeat(embeddings_folder),
                    iter_pair_of_lines,
                    itertools.repeat(sigma),
                    itertools.repeat(mmd_dest_file),
                    itertools.repeat(duration))

    pool.starmap(aux_compute_mmd_between_line, iter_args, chunksize=50)
    pool.close()
    pool.join()
    # for args in iter_args:
    #     aux_compute_mmd_between_line(*args)
if __name__ == '__main__':
    embeddings_folder = '/home/alexandre/workspace/t5_embeddings'
    duration = 10 # corresponds to roughly two seconds of post-activation activity
    mmd_dest_file = '/home/alexandre/workspace/t5_embeddings/mmdtraj{}.csv'.format(duration)

    n_procs = 10

    main(embeddings_folder, duration, mmd_dest_file, n_procs)