import os
import argparse

def aggregate(prefix, root):
    aggregated_stats = dict()
    for fn in os.listdir(root):
        if not(fn.startswith(prefix) and fn.endswith('_bootstrap.csv')):
            continue
        with open(os.path.join(root, fn), 'r') as f:
            lines = [[s.strip() for s in l.split(',')] for l in f.readlines()]
        for line, protocol, pos, total, mmd2 in lines:
            if (line, protocol) in aggregated_stats:
                aggregated_stats[(line, protocol)][0] += int(pos)
                aggregated_stats[(line, protocol)][1] += int(total)
                # assert(aggregated_stats[line][2] == float(mmd2))
            else:
                aggregated_stats[(line, protocol)] = [int(pos), int(total), float(mmd2)]
    with open(f'{prefix}_aggregated_bootstrap.csv', 'w') as f:
        for line, protocol in aggregated_stats:
            pos, total, mmd2 = aggregated_stats[(line, protocol)]
            f.write(f"{line}, {protocol}, {pos}, {total}, {mmd2}\n")

def list_prefixes(root):
    prefixes = set()
    for fn in os.listdir(root):
        if fn.endswith('_bootstrap.csv'):
            prefix = '_'.join(fn.split('_')[:-2])
            prefixes.add(prefix)
    return prefixes

def main(args):
    root = args.root
    prefixes = list_prefixes(root)
    for prefix in prefixes:
        aggregate(prefix, root)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root')
    args = parser.parse_args()
    main(args)
