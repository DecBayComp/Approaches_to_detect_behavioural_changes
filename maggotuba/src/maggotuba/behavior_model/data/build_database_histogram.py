import multiprocessing as mp
import os
import argparse

from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from maggotuba.behavior_model.data.enums import Label, Timeslot, Tracker, Feature

import json

def count_samples(input_queue,\
                  output_queue,\
                  traj_len,\
                  pred_len):
        

    while (args:=input_queue.get()) is not None:
        fn, tracker = args
        data = np.loadtxt(fn)

        # check that there is prestimulus data to renormalize
        after_settled_before_activation = np.logical_and(data[:, Feature.TIME.value]<Tracker.get_stimulus_time(tracker),
                                                         data[:, Feature.TIME.value]>Tracker.get_start_time(tracker))
        if np.sum(after_settled_before_activation) == 0:
            continue

        # count labels
        labels = np.argmax(data[:,1:7], axis=1)
        L = len(data)
        time_slicer = slice(traj_len//2+pred_len,L-traj_len//2-pred_len) # TODO : probably a little false, probably good enough
        counts = np.zeros((len(['t5', 't15']), len(Label), len(Timeslot)), dtype=int)
        for time, label in zip(data[time_slicer,0], labels[time_slicer]):
            timeslot = Timeslot.from_timestamp(time, tracker).value # TODO can be much faster, no need for the for loop ?
            counts[tracker.value, label, timeslot] += 1

        output_queue.put(counts)

def writer_process_target(counts_queue, output_file):
    counts = np.zeros((len(['t5', 't15']), len(Label), len(Timeslot)), dtype=int)
    pbar = tqdm(total=577289)
    while (partial_counts:=counts_queue.get()) is not None:
        counts += partial_counts
        pbar.update()
    print('sum : ', counts)
    np.save(output_file, counts)

def iterfiles(root):
    for tracker in ['t5', 't15']:
        for protocol in os.listdir(os.path.join(root, tracker)):
            for line in os.listdir(os.path.join(root, tracker, protocol)):
                for datetime in os.listdir(os.path.join(root, tracker, protocol, line)):
                    for fn in os.listdir(os.path.join(root, tracker, protocol, line, datetime)):
                        if fn.endswith('.txt') and fn.startswith('Point_dynamics_'):
                            yield os.path.join(root, tracker, protocol, line, datetime, fn), Tracker[tracker.upper()]

def add_arguments(parser):
    parser.add_argument('--n_workers', default=40, type=int)
    parser.add_argument('--source', default=None)
    parser.add_argument('--len_traj', default=None, type=int)
    parser.add_argument('--len_pred', default=None, type=int)
    parser.add_argument('--output_file', default='counts.npy')

def main(args):
    try:
        with open('config.json', 'r') as config_file:
            config = json.load(config_file)
    except FileNotFoundError:
        raise ValueError("Please run from a project folder.")

    if args.source is None:
        args.source = config['raw_data_dir']
    if args.len_traj is None:
        args.len_traj = config['len_traj']
    if args.len_pred is None:
        args.len_pred = config['len_pred']
    

    source = os.path.abspath(args.source)

    print('Counting samples...')

    input_queue = mp.Queue()
    counts_queue = mp.Queue()

    counts_pool = []
    for i in range(args.n_workers):
        counts_pool.append(mp.Process(target=count_samples, args=(input_queue, counts_queue, args.len_traj, args.len_pred)))
    [p.start() for p in counts_pool]

    writer_process = mp.Process(target=writer_process_target, args=(counts_queue, args.output_file))
    writer_process.start()

    for counts_args in iterfiles(source):
        input_queue.put(counts_args)
    for p in counts_pool:
        input_queue.put(None) # send termination flag
    for p in counts_pool:
        p.join()
    
    counts_queue.put(None) # send termination flag
    writer_process.join()    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()
    main(args)