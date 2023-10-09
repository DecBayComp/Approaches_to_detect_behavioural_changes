import os
import itertools
import logging
import multiprocessing as mp
import matplotlib.pyplot as plt

import numpy as np
from maggotuba.mmd import compute_linear_estimator_of_mmd
from scipy.spatial import distance_matrix
from tqdm import tqdm

def load_embedding(embeddings_dest_folder, line_folder, protocol_folder, duration=20, cutoff=None):
    filename = os.path.join(embeddings_dest_folder,
                            line_folder,
                            protocol_folder,
                            'encoded_trajs.npy')
    data = np.load(filename)
    data = data[:,:duration,:]
    if cutoff:
        data = data[:cutoff//duration,:,:]
    return data

def main(embeddings_folder, ref_line, ref_protocol, duration):
    sigma = 5.5

    embedding = load_embedding(embeddings_folder, ref_line, ref_protocol, duration=duration)
    rng = np.random.default_rng()
    ns = [100,200,500,1000,1500,2000]
    means = []
    stds = []
    recordings = []
    plt.figure()

    for n in ns:
        n_recordings = []
        for iter_ in range(1000):
            subset1 = rng.choice(embedding, size=n//duration)
            subset2 = rng.choice(embedding, size=n//duration)
            subset1.reshape(-1, 10)
            subset2.reshape(-1, 10)
            mmd, _, _ = compute_linear_estimator_of_mmd(subset1, subset2, sigma)
            n_recordings.append(mmd)
        means.append(np.mean(n_recordings))
        stds.append(np.std(n_recordings))
        plt.hist(n_recordings, density=True, bins=25)

    plt.legend(labels=ns)
    plt.title('Distribution of $MMD^2$ as a function of sample size - {}/{}'.format(ref_line, ref_protocol))
    plt.show()


    plt.figure()
    plt.plot(ns, means)
    plt.title('Mean as a function of sample_size')
    plt.show()

    plt.figure()
    plt.plot(ns, stds)
    plt.title('Std as a function of sample_size')
    plt.show()

if __name__ == '__main__':
    embeddings_folder = '/home/alexandre/workspace/t5_embeddings'
    ref_line = 'FCF_attP2_1500062@UAS_TNT_2_0003'
    ref_protocol = 'p_8_45s1x30s0s#p_8_105s10x2s10s#n#n@100'
    duration = 20 # corresponds to roughly two seconds of post-activation activity

    main(embeddings_folder, ref_line, ref_protocol, duration)