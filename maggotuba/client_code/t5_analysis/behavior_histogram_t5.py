from tqdm import tqdm
import os
import itertools
import logging

import numpy as np

from maggotuba.behavior_model.data.enums import Tracker
from maggotuba.behavior_model.data.enums import Timeslot
from maggotuba.behavior_model.data.enums import Feature

def compute_histogram(line_root, line_dest_folder, traj_length, start_time):
    os.makedirs(line_dest_folder, exist_ok=True)

    counts = np.zeros(6, dtype=int)
    for root, dirnames, filenames in os.walk(line_root):
        for fn in filenames:
            if fn.endswith('.txt') and fn.startswith('Point_dynamics_'):
                data = np.loadtxt(os.path.join(line_root, root, fn))
                if not sum(data[:,0] >= start_time):
                    logging.debug('no point after start_time')
                    continue
                start_point = np.min(np.argwhere(data[:,0] >= start_time))

                time_selected_data = data[start_point:start_point+traj_length] # see explanations above
                if len(time_selected_data) != traj_length:
                    logging.debug('not enough points')
                    continue

                labels = time_selected_data[:,Feature.FIRST_LABEL.value:Feature.LAST_LABEL.value+1].astype(int)
                counts += np.sum(labels>0, axis=0)
    print(counts)
    np.save(os.path.join(line_dest_folder, f'histogram_{traj_length}.npy'), counts/np.sum(counts))

def iterlines(point_dynamics_root):
    for line in os.listdir(point_dynamics_root):
        if not os.path.isdir(os.path.join(point_dynamics_root, line)):
            continue
        for protocol in os.listdir(os.path.join(point_dynamics_root, line)):
            if (not(os.path.isdir(os.path.join(point_dynamics_root, line, protocol)))
                 or not(protocol.startswith('p_8_45s1x30s0s#p_8_105s10x2s10s#n#n@100')
                     or protocol.startswith('p_3gradient1_45s1x30s0s#p_3gradient1_105s10x2s10s#n#n@100'))):
                continue
            yield line, protocol

def main(point_dynamics_root, histogram_dest_folder, traj_length, start_time):
    # embed the lines
    with tqdm(total=1396) as pbar:
        for line_folder, protocol_folder in iterlines(point_dynamics_root):
            line_root = os.path.join(point_dynamics_root, line_folder, protocol_folder)
            line_dest_folder = os.path.join(histogram_dest_folder, line_folder, protocol_folder)
            if os.path.isfile(os.path.join(line_dest_folder, f'histogram_{traj_length}.npy')):
                pbar.update()
                continue # we do not reembed previously embedded trajectories
            compute_histogram(line_root,
                              line_dest_folder,
                              traj_length,
                              start_time)
            pbar.update()


if __name__ == '__main__':
    point_dynamics_root = '/home/alexandre/workspace/larva_dataset/t5_t15_point_dynamics/t5_t15_point_dynamics_data/t5_point_dynamics/point_dynamics_data'
    histogram_dest_folder = '/home/alexandre/workspace/t5_embeddings/histograms'
    tracker = Tracker.T5
    start_time = Tracker.get_stimulus_time(tracker)
    traj_length = 20 # corresponds to roughly 2 seconds of post-activation activity

    main(point_dynamics_root, histogram_dest_folder, traj_length, start_time)