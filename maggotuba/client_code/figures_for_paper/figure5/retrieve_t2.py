import os
import argparse
from mat73 import loadmat
import numpy as np
import multiprocessing as mp
import pandas as pd

def getColumnsFromMat(mat):
    one_larva = not(isinstance(mat['trx']['t'], list))
    if one_larva:
        n_larvae = 1
        point_dynamics = []
        for larva in range(n_larvae):
            columns = []
            # add time column
            columns.append(np.array(mat['trx']['t'][larva]).reshape(-1,1))

            # add behavior label
            for column_name in ['run', 'cast', 'stop', 'hunch', 'back', 'roll']:
                columns.append(np.array(mat['trx'][column_name][larva]).reshape(-1,1)) 

            # add length
            columns.append(np.array(mat['trx']['larva_length_smooth_5'][larva]).reshape(-1,1))

            # add coordinates
            for coordinate_name in ['_tail', '_neck_down', '_neck', '_neck_top', '_head']:
                for axis in ['x', 'y']:
                    columns.append(np.array(mat['trx'][axis+coordinate_name][larva]).reshape(-1,1))

            point_dynamics.append(np.hstack(columns))

        # extract larva number
        larva_numbers = mat['trx']['numero_larva']

    else:
        n_larvae = len(mat['trx']['t'])
        point_dynamics = []
        for larva in range(n_larvae):
            columns = []
            # add time column
            columns.append(np.array(mat['trx']['t'][larva][0]).reshape(-1,1))

            # add behavior label
            for column_name in ['run', 'cast', 'stop', 'hunch', 'back', 'roll']:
                columns.append(np.array(mat['trx'][column_name][larva][0]).reshape(-1,1)) 

            # add length
            columns.append(np.array(mat['trx']['larva_length_smooth_5'][larva][0]).reshape(-1,1))

            # add coordinates
            for coordinate_name in ['_tail', '_neck_down', '_neck', '_neck_top', '_head']:
                for axis in ['x', 'y']:
                    columns.append(np.array(mat['trx'][axis+coordinate_name][larva][0]).reshape(-1,1))

            point_dynamics.append(np.hstack(columns))

        # extract larva number
        larva_numbers = [l[0] for l in mat['trx']['numero_larva']]

    return point_dynamics, larva_numbers

def extract_and_save(dirpath, dest=None):
    if dest is None:
        dest = dirpath
    mat = loadmat(os.path.join(dirpath, 'trx.mat'))
    point_dynamics, larva_numbers = getColumnsFromMat(mat)

    # create filename
    aux, date = os.path.split(dirpath)
    aux, protocol = os.path.split(aux)
    _, line = os.path.split(aux)

    purged_line = line.replace('@', '_')
    purged_protocol = protocol.split('@')[0]
    basename = '_'.join(['Point_dynamics', 't2', purged_line, purged_protocol, 'larva_id', date, 'larva_number'])
    for larva_number, larva in zip(larva_numbers, point_dynamics):
        os.makedirs(os.path.join(dest, line, protocol, date), exist_ok=True)
        path = os.path.join(dest, line, protocol, date, basename+'_{}.txt'.format(larva_number))
        # print(f'{basename}_{larva_number}', flush=True)
        np.savetxt(path, larva, delimiter='\t', fmt=['%.3f']+6*['%d']+['%.4f']+10*['%.3f'])

if __name__ == '__main__':
    data_dir = '/mnt/hecatonchire/tihana/t2'
    dest = '/home/alexandre/workspace/larva_dataset/point_dynamics_5/t2'
    lines = []
    preds = pd.read_csv(os.path.abspath(os.path.join(os.path.dirname(__file__), 'DF_final_prediction_23_02_t2.csv')), index_col='index')
    for line in preds['Line']:
        line = line.replace('_p','/p').replace('#/p','#p').replace('_ch','/ch').replace('#/ch','#ch').replace('_LexAop','@LexAop').replace('_LexAop','@LexAop').replace('_20XUAS','@20XUAS').replace('_UAS_TNT','@UAS_TNT')
        line, protocol = line.split('/')
        lines.append((line,protocol))

    for line, protocol in lines:
        for dirpath, dirnames, filenames in os.walk(os.path.join(data_dir, line, protocol)):
            for fn in filenames:
                if fn == 'trx.mat':
                    aux, date = os.path.split(dirpath)
                    aux, protocol = os.path.split(aux)
                    _, line = os.path.split(aux)
                    target_dir = os.path.join(dest, line, protocol, date)
                    already_transformed = os.path.isdir(target_dir) and sum([fn2.startswith('Point_dynamics') for fn2 in os.listdir(target_dir)]) > 0
                    if not(already_transformed):
                        print(dirpath)
                        extract_and_save(dirpath, dest)
                    break