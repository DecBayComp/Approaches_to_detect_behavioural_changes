import pickle
import os
import csv

def main(root, ref_dict):
    effectors = set(t[0].split('@')[1] for t in ref_dict)
    bootstraps = [s for s in os.listdir(root) if s.endswith('_aggregated_bootstrap.csv')]
    
    aux_effectors = set()
    for e in effectors:
        if any([bootstrap.split('@')[1].startswith(e) for bootstrap in bootstraps]):
            aux_effectors.add(e)
    effectors = aux_effectors

    for e in effectors:
        selected_bootstraps = [b for b in bootstraps if b.split('@')[1].startswith(e)]
        dest_fn = 'MMD_'+e
        dest_file = open(os.path.join(root, dest_fn), 'w', newline='')
        writer = csv.writer(dest_file)
        writer.writerow(['line/protocol', 'MMD^2', 'p-value', 'bootstrap size'])
        for bootstrap in selected_bootstraps:
            with open(os.path.join(root, bootstrap), 'r', newline='') as f:
                for row in csv.reader(f):
                    line = '/'.join([s.strip() for s in row[:2]])
                    mmd2 = float(row[4])
                    pos, n = [int(s) for s in row[2:4]]
                    p_value = (n-pos+1)/n
                    to_write = [line, mmd2, p_value, n]
                    writer.writerow(to_write)

        dest_file.close()

if __name__ == '__main__':
    root = '/mnt/hecatonchire/alexandre/aggregated_bootstraps'
    with open('/home/alexandre/workspace/structured-temporal-convolution/client_code/all_hits_t5/ref_dict_t5.pickle', 'rb') as f:
        ref_dict = pickle.load(f)
    main(root, ref_dict)