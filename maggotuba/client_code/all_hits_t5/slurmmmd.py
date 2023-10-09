# standard library packages
import os
from pickle import load
from pathlib import Path

# scientific computing packages
import numpy as np

# data science packages
from scipy.spatial import distance_matrix

# homegrown function
from maggotuba.mmd import cbootstrap


def add_argument_find_hits(parser):
    parser.add_argument('--job_id', type=int, default=0)
    parser.add_argument('--total_jobs', type=int, default=1)
    parser.add_argument('--root', default='experiment_1')
    parser.add_argument('--tracker', '-t', default='t5')
    parser.add_argument('--reference', '-ref', default='FCF_attP2_1500062')
    parser.add_argument('--reference_protocol', '-refp', default=None)
    parser.add_argument('--nboot', type=int, default=10)

def find_hits(args):
    # for all lines in tracker with the proper prefix and effector:
    #     for the given protocol:
    #         compute the MMD between the reference and the line
    #         compute a p value using either spectral chi squared or the actual bootstrap
    #         store the tuple p-value, line


    # collect all lines
    print('Starting')
    with open(os.path.join(os.path.dirname(__file__), 'inv_ref_dict_t5.pickle'), 'rb') as f:
        d = load(f)
    collected_lines = d[args.reference, args.reference_protocol]

    # load the list of lines rejected from embedding
    with open(os.path.join(args.root, 'embeddings', args.tracker, 'rejected.txt'), 'r') as f:
        rejected_lines = f.readlines()
        rejected_lines = {tuple(s.strip() for s in l.split()) for l in rejected_lines}

    # exclude rejected lines from collected lines
    for l in collected_lines:
        if l in rejected_lines:
            collected_lines.remove(l)
            print(f"{l} was rejected.")

    # chose lines to deal with
    q, r = len(collected_lines)//args.total_jobs, len(collected_lines)%args.total_jobs
    n_lines = q + (args.job_id < r)
    start = args.job_id*q + min(r, args.job_id)
    collected_lines = collected_lines[start:start+n_lines]
    print(50*'#')
    for t in collected_lines:
        print(' '.join(t))
    print(50*'#')

    # load reference
    print("Loading reference")
    args.reference_protocol = args.reference_protocol if args.reference_protocol is not None else args.protocol
    ref_path = os.path.join(args.root, 'embeddings', args.tracker, args.reference, args.reference_protocol, 'encoded_trajs.npy')
    reference = np.load(ref_path)
    print("Reference loaded. Computing sigma")
    sigma = np.median(distance_matrix(reference, reference))
    print("Sigma : ", sigma)

    #############################################################
    #
    #           TIME EFFICIENT CYTHON BOOTSTRAP !
    #
    ##########################################################

    # process lines and write results to bootstrap.csv
    pvals = []
    for i, (line, protocol) in enumerate(collected_lines):
        print(i, '/', len(collected_lines), ' : ', line)
        line_path = os.path.join(args.root, 'embeddings', args.tracker, line, protocol, 'encoded_trajs.npy')
        data = np.load(line_path)
 
        print("bootstrap...")
        testStat, bootstrap = cbootstrap(data, reference, sigma, args.nboot)
        print("Sorting and placing pval...")
        bootstrap = np.sort(bootstrap)
        pos = np.searchsorted(bootstrap, testStat)
        print("Writing...")
        with open(os.path.join(args.root, f'{args.reference}_{args.reference_protocol}_{args.job_id}_bootstrap.csv'), 'a') as f:
            f.write(f"{line}, {protocol}, {pos}, {args.nboot}, {testStat}\n")
        print("done.")

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    add_argument_find_hits(parser)
    args = parser.parse_args()
    find_hits(args)
