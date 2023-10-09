# standard library packages
import os
from pickle import load

# scientific computing packages
import numpy as np

# data science packages
from scipy.spatial import distance_matrix



def add_argument_compute_print_sigmas(parser):
    parser.add_argument('--root', default='experiment_1')
    parser.add_argument('--tracker', '-t', default='t5')


def compute_print_sigmas(args):
    # collect all lines
    print('Starting')
    with open(os.path.join(os.path.dirname(__file__), 'inv_ref_dict_t5.pickle'), 'rb') as f:
        d = load(f)

    for reference_line, reference_protocol in d:
        ref_path = os.path.join(args.root, 'embeddings', args.tracker, reference_line, reference_protocol, 'encoded_trajs.npy')
        reference = np.load(ref_path)
        sigma = np.median(distance_matrix(reference, reference))
        print(f"{reference_line}, {reference_protocol}, {sigma}")


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    add_argument_compute_print_sigmas(parser)
    args = parser.parse_args()
    compute_print_sigmas(args)
