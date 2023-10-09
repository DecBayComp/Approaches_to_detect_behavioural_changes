import os
import itertools
import logging
import multiprocessing as mp

import numpy as np
from maggotuba.mmd import compute_linear_estimator_of_mmd, compute_quadratic_estimator_of_mmd
from maggotuba.mmd import rbf_dot
from scipy.spatial import distance_matrix
from tqdm import tqdm

import tempfile

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
    return data.reshape(-1, data.shape[-1])

def compute_deciles_one_line(embeddings_folder, line_folder, protocol_folder, duration=20):
        embedding = load_embedding(embeddings_folder, line_folder, protocol_folder, duration=duration)
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

def compute_mmd_between_line(dot_product_dict, dot_square_dict, embeddings_folder, pair_of_lines, sigma, mmd_dest_file, duration):
        line1_protocol1, line2_protocol2 = pair_of_lines

        mmd = dot_square_dict[line1_protocol1] + dot_square_dict[line2_protocol2] - dot_product_dict[pair_of_lines]
        with open(mmd_dest_file, 'a') as f:
            f.write("{}, {}, {}, {}, {}\n".format(line1, protocol1,
                                                  line2, protocol2,
                                                  mmd))

def compute_dot_square(dot_square_dict, embeddings_folder, line_protocol, sigma, duration=20):
    line, protocol = line_protocol
    X = load_embedding(embeddings_folder, line, protocol, duration=duration, cutoff=20000)
    m = len(X)
    if m > 10000:
        print('computing', 'len = ', m)
    dot_square = 1/(m*(m-1))*np.sum(np.triu(rbf_dot(X, X, sigma), k=1))
    dot_square_dict[line_protocol] = dot_square

def compute_dot_products(dot_product_dict, embeddings_folder, lines_protocols_pair, sigma, duration=20):
    lines_protocols1, lines_protocols2 = lines_protocols_pair
    data_dict = {}
    for line, protocol in itertools.chain(*lines_protocols_pair):
        data_dict[(line, protocol)] = load_embedding(embeddings_folder, line, protocol, duration=duration, cutoff=20000)
    tot = len(lines_protocols1)*len(lines_protocols2)
    i=0
    for line1, protocol1 in lines_protocols1:
        for line2, protocol2 in lines_protocols2:
            if line1 == line2 and protocol1 == protocol2:
                continue
            if ((line1, protocol1), (line2, protocol2)) in dot_product_dict.keys():
                continue
            print('{}/{}'.format(i, tot))
            X, Y = data_dict[(line1, protocol1)], data_dict[(line2, protocol2)]
            C = rbf_dot(X, Y, sigma)
            m, n = len(X), len(Y)
            mmd = 2/m/n*np.sum(C)
            dot_product_dict[((line1, protocol1), (line2, protocol2))] = mmd
            dot_product_dict[((line2, protocol2), (line1, protocol1))] = mmd
            i+=1




    

def main(embeddings_folder, duration, mmd_dest_file, n_procs):
    # compute the kernel size
    # compute deciles
    # deciles = []
    # N = []
    # with tqdm(total=31) as pbar:
        # for line_folder, protocol_folder in iterlines(embeddings_folder):
            # if line_folder.startswith('FCF_'):
                # deciles_line, N_line = compute_deciles_one_line(embeddings_folder, line_folder, protocol_folder, duration=duration)
                # deciles.append(deciles_line)
                # N.append(N_line)
                # pbar.update()

    # # aggregate deciles to compute median
    # deciles = np.vstack(deciles)
    # N = np.array(N)
    # sigma = aggregate_deciles(N, deciles)
    # print('sigma : ', sigma)
    sigma = 5.496 # for 20 time points

    pool = mp.Pool(n_procs)
    manager = mp.Manager()
    dot_square = manager.dict()
    dot_product = manager.dict()

    # compute the scalar squares
    print("computing squares...")
    iter_args = zip(itertools.repeat(dot_square),
                    itertools.repeat(embeddings_folder),
                    iterlines(embeddings_folder),
                    itertools.repeat(sigma),
                    itertools.repeat(duration))
    pool.starmap(compute_dot_square, iter_args, chunksize=50)
    pool.close()
    pool.join()
    pool.terminate()

    # compute the dot products
    # we split the upper triangle in roughly rectangular blocks in order to reduce I/O
    pool = mp.Pool(n_procs)

    print("computing_products...")
    lines_protocols = list(iterlines(embeddings_folder))
    lines_protocols_pairs = []
    N = len(lines_protocols)
    block_size = 50
    n_blocks = N//block_size + 1
    for i in range(n_blocks):
        for j in range(i+1, n_blocks):
            lines_protocols_pairs.append((lines_protocols[i*block_size:(i+1)*block_size], lines_protocols[j*block_size:(j+1)*block_size]))

    iter_args = zip(itertools.repeat(dot_product),
                itertools.repeat(embeddings_folder),
                lines_protocols_pairs,
                itertools.repeat(sigma),
                itertools.repeat(duration))
    pool.starmap(compute_dot_products, iter_args, chunksize=10)
    pool.close()
    pool.join()
    pool.terminate()


    # compute the mmd
    pool = mp.Pool(n_procs)

    iter_pair_of_lines = itertools.permutations(iterlines(embeddings_folder), r=2)
    iter_args = zip(itertools.repeat(dot_product),
                    itertools.repeat(dot_square),
                    itertools.repeat(embeddings_folder),
                    iter_pair_of_lines,
                    itertools.repeat(sigma),
                    itertools.repeat(mmd_dest_file),
                    itertools.repeat(duration))

    pool.starmap(compute_mmd_between_line, iter_args, chunksize=50)
    pool.close()
    pool.join()
    pool.terminate()

if __name__ == '__main__':
    embeddings_folder = '/home/alexandre/workspace/t5_embeddings'
    duration = 20 # corresponds to roughly two seconds of post-activation activity
    mmd_dest_file = '/home/alexandre/workspace/t5_embeddings/mmd{}_quadratic.csv'.format(duration)

    n_procs = 10

    main(embeddings_folder, duration, mmd_dest_file, n_procs)