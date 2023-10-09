import multiprocessing as mp
import os.path
import numpy.random as rnd
import numpy as np
import argparse
import h5py
import itertools
from datetime import datetime
import tqdm

from maggotuba.behavior_model.data.enums import Label, Timeslot, Tracker, Feature
from maggotuba.behavior_model.data.utils import center_coordinates, rotate, compute_rotation_matrix

import json
# ################################################################################
# #                                                                              #          
# #                            HARDCODED REJECTION SAMPLING                      #           
# #                                                                              #           
# ################################################################################

# #TODO Refactor to avoid hardcoded values

# # sample count obtained using exhaustive_sample_counting.py
# # hardcoded for simplicity
# # dimensions : {t5, t15} x {run, bend, stop, hunch, back, roll} x {setup, before, during, after}
# N_SAMPLES_NOT_SCALED = np.array([[[ 52131648,  23265657,   2624668, 303889498],
#                                   [ 18587317,   9011929,   2650373, 228331523],
#                                   [  1069264,    763524,    469436,  34341526],
#                                   [    23975,     34445,    733270,   5577181],
#                                   [   502655,    327770,    727195,  13387502],
#                                   [    51263,     26229,      6565,    430490]],
                     
#                                  [[ 16790227,  19938756,   3023452, 172225718],
#                                   [  5541839,   7453778,   2641581,  90876749],
#                                   [    66120,    168406,    160658,   6385043],
#                                   [    19237,     28588,     51414,    346758],
#                                   [    53589,    100456,     77792,   2046120],
#                                   [     4553,      6255,     21254,    296754]]])

# N_SAMPLES_SCALED = np.array([[[ 44265378,  23287124,   2626831,  99076911],
#                               [ 16087105,   9017810,   2653400, 104674121],
#                               [   983756,    763512,    469672,  19415447],
#                               [    20444,     34455,    734542,   1924362],
#                               [   441881,    328837,    727931,   8786846],
#                               [    43568,     26233,      6575,    168208]],
                      
#                              [[ 16777787,  19962812,   3023203, 113832211],
#                               [  5492955,   7459428,   2639699,  60492007],
#                               [    66324,    168762,    160765,   4181698],
#                               [    18907,     28604,     51429,    224825],
#                               [    53597,    100659,     77818,   1524450],
#                               [     4528,      6253,     21240,    205598]]])


# N_SAMPLES = N_SAMPLES_SCALED


# TARGET_SAMPLES = 100000               # to create enough samples use TARGET_SAMPLES = 100000
# TOTAL  = np.sum(N_SAMPLES)

# def compute_selection_rate(kw, TARGET_SAMPLES):
#     PROPORTIONS = N_SAMPLES/np.sum(N_SAMPLES)

#     TARGET_PROPORTIONS = N_SAMPLES.copy()
#     # do not sample from the 'setup' timeslot
#     TARGET_PROPORTIONS[:,:,Timeslot.SETUP.value] = 0.0
#     # do not sample rolls from t5
#     TARGET_PROPORTIONS[Tracker.T5.value,Label.ROLL.value,:] = 0.0
#     # normalize behavior-wise
#     TARGET_PROPORTIONS = TARGET_PROPORTIONS/np.sum(np.sum(TARGET_PROPORTIONS, axis=0 , keepdims=True), axis=2, keepdims=True)
#     # behavior weights
#     if kw == 'balanced':
#         BEHAVIOR_WEIGHTS = np.array([0.25, 0.25, 0.125, 0.125, 0.125, 0.125]).reshape(1,6,1)
#     if kw == 'run-bend':
#         BEHAVIOR_WEIGHTS = np.array([0.5, 0.5, 0., 0., 0., 0.]).reshape(1,6,1)

#     # allocate weights to behaviors
#     TARGET_PROPORTIONS = BEHAVIOR_WEIGHTS*TARGET_PROPORTIONS
    
#     if TARGET_SAMPLES == 'max':
#         SELECTION_PROPORTION = int(1./np.min(TARGET_PROPORTIONS/PROPORTIONS))
#     else:
#         SELECTION_PROPORTION = TARGET_SAMPLES/TOTAL
#     SELECTION_RATE = SELECTION_PROPORTION*TARGET_PROPORTIONS/PROPORTIONS
#     return SELECTION_RATE

# SELECTION_RATE = compute_selection_rate('balanced', TARGET_SAMPLES)
# assert(np.all(SELECTION_RATE <= 1.0))

# ###############################################################################
# #
# #              END OF HARDCODED REJECTION SAMPLING
# #
# ################################################################################

def compute_selection_rate(counts, target_samples, kind='balanced'):
    proportions = counts/np.sum(counts)

    target_proportions = counts.copy()
    # do not sample from the 'setup' timeslot
    target_proportions[:,:,Timeslot.SETUP.value] = 0.0
    # do not sample rolls from t5
    target_proportions[Tracker.T5.value,Label.ROLL.value,:] = 0.0
    # normalize behavior-wise
    target_proportions = target_proportions/np.sum(np.sum(target_proportions, axis=0 , keepdims=True), axis=2, keepdims=True)
    # behavior weights
    if kind == 'balanced':
        behavior_weights = np.array([0.25, 0.25, 0.125, 0.125, 0.125, 0.125]).reshape(1,6,1)
    if kind == 'run-bend':
        behavior_weights = np.array([0.5, 0.5, 0., 0., 0., 0.]).reshape(1,6,1)

    # allocate weights to behaviors
    target_proportions = behavior_weights*target_proportions
    
    if target_samples == 'max':
        selection_proportions = int(1./np.min(target_proportions/proportions))
    else:
        selection_proportions = target_samples/np.sum(counts)
    selection_rate = selection_proportions*target_proportions/proportions
    return selection_rate

def writer_process_target(filename, queue, len_traj, len_pred, target_samples):

    f = h5py.File(filename, 'w')
    group = f.create_group('samples')
    group.attrs['len_pred'] = len_pred
    group.attrs['len_traj'] = len_traj
    group.attrs['registered'] = 1
    group.attrs['rescaled'] = 'prestimulus'
    sample_idx = 0
    pbar = tqdm.tqdm(total=target_samples)
    while (to_write := queue.get()) is not None:
        data, tracker, line, protocol, datetime, larva_number, filename, timeslot, start_point, behavior, path = to_write 
        dset = f.create_dataset(f'samples/sample_{sample_idx}', data.shape, dtype=data.dtype)
        dset[...] = data
        dset.attrs['tracker']      = tracker
        dset.attrs['line']         = line
        dset.attrs['protocol']     = protocol
        dset.attrs['datetime']     = datetime
        dset.attrs['larva_number'] = larva_number
        dset.attrs['filename']     = filename
        dset.attrs['timeslot']     = timeslot
        dset.attrs['start_point']  = start_point
        dset.attrs['behavior']     = behavior
        dset.attrs['path']         = path 

        sample_idx += 1
        pbar.update()
    group.attrs['n_samples'] = sample_idx
    f.close()

def filereader_process_target(input_queue, output_queue, len_traj, len_pred, selection_rates):
    rng = rnd.default_rng()
    while (fn := input_queue.get()) is not None:
        tracker = Tracker.from_path(fn)
        data = np.loadtxt(fn)

        # parse filename
        dirname, filename = os.path.split(fn)
        dirname, datetime = os.path.split(dirname)
        dirname, protocol = os.path.split(dirname)
        dirname, line     = os.path.split(dirname)
        larva_number = int((os.path.splitext(filename)[0]).split('_')[-1])

        # Normalize length according to mean length before stimulus
        after_settled_before_activation = np.logical_and(data[:, Feature.TIME.value]<Tracker.get_stimulus_time(tracker),
                                                        data[:, Feature.TIME.value]>Tracker.get_start_time(tracker))
        if np.sum(after_settled_before_activation) == 0:
            # No prestimulus data, the file is discarded
            continue
        mean_larva_length = np.mean(data[after_settled_before_activation, Feature.LENGTH.value])
        data[:, Feature.LENGTH.value:Feature.LAST_COORD.value+1] = data[:, Feature.LENGTH.value:Feature.LAST_COORD.value+1]/mean_larva_length

        # iterate across all windows
        window_length = 2*len_pred+len_traj
        midpoint = window_length//2-1 if window_length %2 else (window_length-1)//2-1

        try:
            for start_point in range(len(data)-window_length+1):
                sample = data[start_point:start_point+window_length]
                label = np.argmax(sample[midpoint, Feature.FIRST_LABEL.value:Feature.LAST_LABEL.value+1]).item()
                center_time = sample[midpoint,Feature.TIME.value]
                timeslot = Timeslot.from_timestamp(center_time, tracker)

                # perform sample rejection to balance the dataset
                if rng.random() > selection_rates[tracker.value, label, timeslot.value]:
                    continue

                # rotate the sample based on present values
                sample = sample.copy()
                sample = rotate(sample, slice(len_pred, len_pred+len_traj))

                # center the samples based on present values
                sample = center_coordinates(sample, slice(len_pred, len_pred+len_traj))

                output_queue.put((sample, tracker.name, line, protocol, datetime, larva_number, filename, timeslot.name, start_point, Label(label).name, fn))

        except ValueError as e:
            if str(e) == 'ValueError: window shape cannot be larger than input array shape':
                print('Not enough points to form a window')
            else:
                raise e    

def iterfiles(root):
    for tracker in ['t5', 't15']:
        for protocol in os.listdir(os.path.join(root, tracker)):
            for line in os.listdir(os.path.join(root, tracker, protocol)):
                for datetime in os.listdir(os.path.join(root, tracker, protocol, line)):
                    for fn in os.listdir(os.path.join(root, tracker, protocol, line, datetime)):
                        if fn.endswith('.txt') and fn.startswith('Point_dynamics_'):
                            yield os.path.join(root, tracker, protocol, line, datetime, fn)

def main(args):
    # load counts
    try:
        counts = np.load(args.counts_file)
    except FileNotFoundError:
        raise FileNotFoundError("Please run 'maggotuba db counts' beforehand.")

    selection_rates = compute_selection_rate(counts, args.n_samples, kind='balanced')
    assert np.all(selection_rates <= 1.0), f'Selection rates should all be smaller than 1. Instead, selection rates are : \n {selection_rates}'

    # load config
    try:
        with open('config.json', 'r') as config_file:
            config = json.load(config_file)
    except FileNotFoundError:
        raise ValueError("Please provide a target directory or run from a project folder.")

    if args.len_traj is None:
        args.len_traj = config['len_traj']
    if args.len_pred is None:
        args.len_pred = config['len_pred']

    # build source from config file
    if args.source is None:            
        args.source = config['raw_data_dir']
    args.source = os.path.abspath(args.source)
    
    # build target from config file
    if args.target is None:
        if config == None:
            try:
                with open('config.json', 'r') as config_file:
                    config = json.load(config_file)
            except FileNotFoundError:
                raise ValueError("Please provide a target directory or run from a project folder.")
        date = datetime.now().strftime('%Y_%m_%d')
        filename = f'larva_dataset_{date}_{args.len_traj}_{args.len_pred}_{args.n_samples}.hdf5'
        filepath = os.path.join(config['project_dir'], filename)
        args.target = filepath
    args.target = os.path.abspath(args.target)

    # set database file in config to target
    with open('config.json', 'r') as config_file:
        config = json.load(config_file)
    config['data_dir'] = args.target
    with open('config.json', 'w') as config_file:
        config = json.dump(config, config_file)

    input_queue = mp.Queue()
    output_queue = mp.Queue()
    
    pool = [mp.Process(target=filereader_process_target, args=(input_queue, output_queue,args.len_traj,args.len_pred,selection_rates)) 
            for _ in range(args.n_workers)]
    for p in pool:
        p.start()

    writer_process = mp.Process(target=writer_process_target, args=(args.target, output_queue,args.len_traj,args.len_pred, args.n_samples))
    writer_process.start()

    for fn in iterfiles(args.source):
        input_queue.put(fn)
    for p in pool:
        input_queue.put(None) # send termination flag
    for p in pool:
        p.join()

    output_queue.put(None) # send termination flag
    writer_process.join()

    # print stats about the dataset
    with h5py.File(args.target, 'r') as f:
        counts = 6*[0]
        mapping = {'RUN':0, 'BEND':1, 'STOP':2, 'HUNCH':3, 'BACK':4, 'ROLL':5}
        for sample in f['samples']:
            counts[mapping[f['samples'][sample].attrs['behavior']]] += 1
        total = sum(counts)
        print('Total : ', total)
        print(' '.join([f'{k.lower()} : {c/total*100:.1f}' for k, c in zip(mapping, counts)])) 

def add_arguments(parser):
    parser.add_argument('--n_workers', default=40, type=int)
    parser.add_argument('--source', default=None, type=str)
    parser.add_argument('--target', default=None, type=str)
    parser.add_argument('--len_traj', default=None, type=int)
    parser.add_argument('--len_pred', default=None, type=int)
    # TODO change this to load config

    parser.add_argument('--n_samples', default=100000, type=int)
    parser.add_argument('--counts_file', default='counts.npy', type=str)

#############################################################################
#
#               Build an outline database            
#
################################################################################
def get_transformation(path, start_point, len_pred, len_traj):
    tracker = Tracker.from_path(path)
    data = np.loadtxt(path)
    # Normalize length according to mean length before stimulus
    after_settled_before_activation = np.logical_and(data[:, Feature.TIME.value]<Tracker.get_stimulus_time(tracker),
                                                    data[:, Feature.TIME.value]>Tracker.get_start_time(tracker))
    mean_larva_length = np.mean(data[after_settled_before_activation, Feature.LENGTH.value])

    sample = data[start_point:start_point+2*len_pred+len_traj].copy()
    sample[:,Feature.FIRST_COORD.value:Feature.LAST_COORD.value+1] /= mean_larva_length

    coords = sample[:,Feature.FIRST_COORD.value:Feature.LAST_COORD.value+1]
    matrix = compute_rotation_matrix(coords[len_pred:len_pred+len_traj])

    sample = rotate(sample, slice(len_pred,len_pred+len_traj))
    bias_x = np.mean(sample[len_pred:len_pred+len_traj,Feature.X_MID_SEGMENT.value])
    bias_y = np.mean(sample[len_pred:len_pred+len_traj,Feature.Y_MID_SEGMENT.value])

    return mean_larva_length, matrix, np.array([bias_x, bias_y])
def fetch_and_write_outline(idx, path, start_point, len_pred, len_traj, dest_file, root='/mnt/hecatonchire/screens'):
    from scipy.fft import rfft

    # example path : /home/alexandre/workspace/larva_dataset/point_dynamics_5/t5/pBDPGAL4U@UAS_TNT_2_0003/p_8_45s1x30s0s#p_8_105s10x2s10s#n#n@100/20110321_110117/Point_dynamics_t5_pBDPGAL4U_UAS_TNT_2_0003_p_8_45s1x30s0s#p_8_105s10x2s10s#n#n_larva_id_20110321_110117_larva_number_545.txt
    # construct path to outlines
    aux, filename = os.path.split(path)
    aux, date = os.path.split(aux)
    aux, protocol = os.path.split(aux)
    aux, line = os.path.split(aux)
    _, tracker = os.path.split(aux)

    path_to_outlines = os.path.join(root, tracker, line, protocol, date)
    path_to_outlines = os.path.join(path_to_outlines, 'trx.mat')

    # get larva number
    larva_number = int(filename[filename.find('larva_number_'):][13:-4])

    # read the outlines
    try:
        f = h5py.File(path_to_outlines, 'r')
        larva_numbers = [int(f[ref][0,0]) for ref in f['trx/numero_larva_num'][0,:]]
        larva_idx_in_file = larva_numbers.index(larva_number)
        outline_x = f[f['trx/x_contour'][0,larva_idx_in_file]][:,start_point:start_point+2*len_pred+len_traj]
        outline_y = f[f['trx/y_contour'][0,larva_idx_in_file]][:,start_point:start_point+2*len_pred+len_traj]
        times = f[f['trx/t'][0,larva_idx_in_file]][:,start_point:start_point+2*len_pred+len_traj].flatten()
        found = True
        f.close()

    except (OSError, AttributeError):
        times = np.zeros(2*len_pred+len_traj)
        specs_x = np.full((2*len_pred+len_traj,13), np.nan, dtype=complex)
        specs_y = np.full((2*len_pred+len_traj,13), np.nan, dtype=complex)
        found = False

    if found:
        scale, matrix, bias = get_transformation(path, start_point, len_pred, len_traj)
        outline_x /= scale
        outline_y /= scale
        outlines = np.stack([outline_x, outline_y], axis=-1)
        outlines = np.einsum('ji,tpi->tpj', matrix, outlines) - bias.reshape(1,1,-1)
        outline_x, outline_y = outlines[:,:,0], outlines[:,:,1]

        specs_x = np.empty((2*len_pred+len_traj,13),dtype=complex)
        for t, x_at_t in enumerate(outline_x.transpose()):
            x_at_t = x_at_t[np.logical_not(np.isnan(x_at_t))]
            spec = rfft(x_at_t)
            spec = spec[:13]/len(x_at_t)
            spec = np.pad(spec, (0, max(0, 13-len(spec))))
            specs_x[t,:] = spec

        specs_y = np.empty((2*len_pred+len_traj,13),dtype=complex)
        for t, y_at_t in enumerate(outline_y.transpose()):
            y_at_t = y_at_t[np.logical_not(np.isnan(y_at_t))]
            spec = rfft(y_at_t)
            spec = spec[:13]/len(y_at_t)
            spec = np.pad(spec, (0, max(0, 13-len(spec))))
            specs_y[t,:] = spec

    with h5py.File(dest_file, 'a') as f:
        f.create_group(f'outlines/outline_{idx}')
        f[f'outlines/outline_{idx}'].attrs['valid'] = found

        f.create_dataset(f'outlines/outline_{idx}/fourier_x', specs_x.shape, complex)
        f[f'outlines/outline_{idx}/fourier_x'][...] = specs_x
        f.create_dataset(f'outlines/outline_{idx}/fourier_y', specs_y.shape, complex)
        f[f'outlines/outline_{idx}/fourier_y'][...] = specs_y


def main_outlines(args):
    if args.db is None:
        files = [s for s in os.listdir(os.getcwd()) if s.startswith('larva_dataset_')
                                                   and s.endswith('.hdf5')
                                                   and not(s.endswith('_outlines.hdf5'))]
        assert(len(files) == 1)
        args.db = files[0]
    if args.target is None:
        name, ext = os.path.splitext(args.db)
        args.target = name +'_outlines' + ext
    if args.source is None:
        raise ValueError("Please provide a source folder for outlines.")

    # load config
    with open('config.json', 'r') as config_file:
        config = json.load(config_file)
    len_pred, len_traj = config['len_pred'], config['len_traj']

 
    with h5py.File(args.target, 'w') as f:
        f.create_group('outlines')

    with h5py.File(args.db, 'r') as samples_db:
        n_samples = samples_db['samples'].attrs['n_samples']
        paths = [samples_db[f'samples/sample_{idx}'].attrs['path'] for idx in range(n_samples)]
        start_points = [samples_db[f'samples/sample_{idx}'].attrs['start_point'] for idx in range(n_samples)]

    if args.n_workers > 1:
        with mp.Pool(args.n_workers) as pool:
            pool.starmap(fetch_and_write_outline,
                         zip(range(n_samples),
                                   paths,
                                   start_points,
                                   itertools.repeat(len_pred),
                                   itertools.repeat(len_traj),
                                   itertools.repeat(args.target),
                                   itertools.repeat(args.source)),
                         chunksize=n_samples//args.n_workers)
    else:
        for i, path, start in tqdm.tqdm(zip(range(n_samples), paths, start_points), total=n_samples):
            fetch_and_write_outline(i, path, start, len_pred, len_traj, args.target, args.source)




def add_arguments_outlines(parser):
    parser.add_argument('--db', default=None, type=str)
    parser.add_argument('--target', default=None, type=str)
    parser.add_argument('--source', default='/mnt/hecatonchire/screens', type=str)
    parser.add_argument('--n_workers', default=1, type=int)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()

    main(args)