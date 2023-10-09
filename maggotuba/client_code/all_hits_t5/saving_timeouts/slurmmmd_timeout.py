# standard library packages
import os
from pickle import load
from pathlib import Path
from time import time
# scientific computing packages
import numpy as np

# data science packages
from scipy.spatial import distance_matrix

# homegrown function
from maggotuba.mmd import cbootstrap



def add_argument_find_hits(parser):
    parser.add_argument('line', default=None)
    parser.add_argument('protocol', default=None)
    parser.add_argument('n_tasks', type=int, default=1)
    parser.add_argument('task_id', type=int, default=0)
    parser.add_argument('--root', default='experiment_1')
    parser.add_argument('--tracker', '-t', default='t5')
    parser.add_argument('--nboot', type=int, default=10)

def find_hits(args):
    # for all lines in tracker with the proper prefix and effector:
    #     for the given protocol:
    #         compute the MMD between the reference and the line
    #         compute a p value using either spectral chi squared or the actual bootstrap
    #         store the tuple p-value, line


    # load reference
    with open(os.path.join(os.path.dirname(__file__), 'ref_dict_t5.pickle'), 'rb') as f:
        d = load(f)
    reference = d[args.line, args.protocol]


    ref_path = os.path.join(args.root, 'embeddings', args.tracker, reference[0], reference[1], 'encoded_trajs.npy')
    ref_data = np.load(ref_path)
    sigma = np.median(distance_matrix(ref_data, ref_data))

    idx = args.task_id
    nboot = (args.nboot//args.n_tasks) + (idx < (args.nboot % args.n_tasks))


    #############################################################
    #
    #           TIME EFFICIENT CYTHON BOOTSTRAP !
    #
    ##########################################################

    # process lines and write results to bootstrap.csv
    pvals = []
    
    line_path = os.path.join(args.root, 'embeddings', args.tracker, args.line, args.protocol, 'encoded_trajs.npy')
    data = np.load(line_path)
    start = time()
    print(line_path, ref_path, sigma)
    print(len(data), len(ref_data))

    testStat, bootstrap = cbootstrap(data, ref_data, sigma, nboot)

    bootstrap = np.sort(bootstrap)
    pos = np.searchsorted(bootstrap, testStat)
    print(time()-start, 's')
    with open(os.path.join(args.root, f'{reference[0]}_{reference[1]}_{idx}_{args.n_tasks}_timeout_bootstrap.csv'), 'a') as f:
        f.write(f"{args.line}, {args.protocol}, {pos}, {nboot}, {testStat}\n")

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    add_argument_find_hits(parser)
    args = parser.parse_args()
    find_hits(args)
