import os
from argparse import ArgumentParser
from pickle import load
from pathlib import Path

def main(args):
    tuples = set()

    with open(os.path.join(os.path.dirname(__file__), 'inv_ref_dict_t5.pickle'), 'rb') as f:
        d = load(f)
    

    with open(args.dest, 'w') as f:
        for t in d:
            f.write(' '.join(t)+'\n')

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dest', default="/home/alexandre/workspace/protocol_list.tmp", type=Path)
    args = parser.parse_args()
    main(args)