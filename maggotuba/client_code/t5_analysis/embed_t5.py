from tqdm import tqdm
import os
import itertools
import logging

import torch
import numpy as np

from maggotuba.behavior_model.args import make_args
from maggotuba.behavior_model.models.model import Trainer
from maggotuba.behavior_model.data.enums import Tracker
from maggotuba.behavior_model.data.enums import Timeslot
from maggotuba.behavior_model.data.enums import Feature
import maggotuba.behavior_model.data.utils as data_utils
from maggotuba.behavior_model.data.datarun import compute_R, rotate

from maggotuba.mmd import compute_linear_estimator_of_mmd

from scipy.spatial import distance_matrix

def load_model(kwargs):
    print('----------ARGUMENTS----------')
    torch.manual_seed(kwargs['seed'])

    model_params = torch.load(kwargs['eval_saved_model']+'/best_validated_model.pt')
    former_kwargs = torch.load(kwargs['eval_saved_model']+'/params.pt')
    path_saved_embedder = os.path.join(kwargs['log_dir'], 'umap/parametric_umap')
    for k, v in kwargs.items():
        former_kwargs[k] = v
    kwargs = former_kwargs
    print('Keyword arguments : ')
    for k, v in kwargs.items():
        print('{} : {}'.format(k, v), flush=True)

    trainer = Trainer(**kwargs)
    trainer.load_state_dict(model_params)

    return trainer

def embed_line(line_root, line_dest_folder, encoder, traj_length, start_time, tracker, window_length, n_features, renormalization_scheme='before-length'):
    os.makedirs(line_dest_folder, exist_ok=True)
    prewindow = window_length//2
    postwindow = window_length-prewindow
    # prewindow points are put before the start point. the window is completed by postwindow points.
    # the center is the first of those postwindow points.

    if renormalization_scheme == 'before-length':
        avg_before_length = compute_before_length(line_root, tracker)
        if avg_before_length is None:
            # there is no way to renormalize the data, unless we normalize it by 1 ?
            folder, protocol = os.path.split(line_dest_folder)
            folder, line = os.path.split(folder)
            with open(os.path.join(folder, 'rejected.txt'), 'a') as f:
                f.write('{} {}\n'.format(line, protocol))
            print("INFO : Skipping {} {} : no renormalization possible".format(line, protocol))
            return

    encoded_trajs = []
    for root, dirnames, filenames in os.walk(line_root):
        for fn in filenames:
            if fn.endswith('.txt') and fn.startswith('Point_dynamics_'):
                data = np.loadtxt(os.path.join(line_root, root, fn))
                if not sum(data[:,0] >= start_time):
                    logging.debug('no point after start_time')
                    continue
                start_point = np.min(np.argwhere(data[:,0] >= start_time))

                time_selected_data = data[start_point-prewindow:start_point+traj_length+postwindow-1] # see explanations above
                if len(time_selected_data) != traj_length + prewindow + postwindow-1:
                    logging.debug('not enough points')
                    continue

                # transform the trajectory to a batch
                windowed_input = np.squeeze(np.lib.stride_tricks.sliding_window_view(time_selected_data, (window_length,time_selected_data.shape[1]))).copy()

                # preprocess the batch
                preprocessed_input = []
                present_indices = slice(prewindow, prewindow+traj_length)
                for i, w in enumerate(windowed_input):
                    # center
                    w = data_utils.center_coordinates(w, present_indices)
                    # rescale
                    if renormalization_scheme == 'traj-length':
                        w = data_utils.rescale_coordinates(w, present_indices)
                    elif renormalization_scheme == 'before-length':
                        is_before = [Timeslot.from_timestamp(row[0], tracker) == Timeslot.BEFORE for row in data]
                        before_data = data[is_before]
                        if len(before_data):
                            w[Feature.FIRST_COORD.value:Feature.LAST_COORD.value+1] /= np.mean(before_data[:,Feature.LENGTH.value])
                        else:
                            w[Feature.FIRST_COORD.value:Feature.LAST_COORD.value+1] /= avg_before_length
                    # select coordinates
                    w = data_utils.select_coordinates_columns(w)
                    w = data_utils.reshape(w)
                    preprocessed_input.append(w)

                input_ = torch.from_numpy(np.stack(preprocessed_input))
                # rotate the batch
                R = compute_R(input_[:,:,:,present_indices])
                input_ = rotate(input_, R)

                # convert to float to run through network
                input_ = input_.float()

                # compute the codes
                with torch.no_grad():
                    output_ = encoder(input_)
                # record
                encoded_trajs.append(output_.numpy())
    encoded_trajs = np.stack(encoded_trajs)
    np.save(os.path.join(line_dest_folder, f'encoded_trajs_{traj_length}.npy'), encoded_trajs)

def compute_before_length(line_root, tracker):
    s = 0
    n = 0
    for root, dirnames, filenames in os.walk(line_root):
        for fn in filenames:
            if fn.endswith('.txt') and fn.startswith('Point_dynamics_'):
                data = np.loadtxt(os.path.join(line_root, root, fn))
                is_before = [Timeslot.from_timestamp(row[0], tracker) == Timeslot.BEFORE for row in data]
                after_setup_before_activation_data = data[is_before]
                if len(after_setup_before_activation_data):
                    after_setup_before_activation_length = np.mean(after_setup_before_activation_data[:, Feature.LENGTH.value])
                    s += after_setup_before_activation_length
                    n += 1

    return s/n if n > 0 else None

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

def main(model_path, point_dynamics_root, embeddings_dest_folder, traj_length, start_time, tracker):
    model_args = make_args(train=False)[0]
    model_args['eval_saved_model'] = model_path
    model_args['log_dir'] = model_path
    trainer = load_model(model_args)
    encoder = trainer.autoencoder.encoder
    encoder.eval()

    # embed the lines
    with tqdm(total=1396) as pbar:
        for line_folder, protocol_folder in iterlines(point_dynamics_root):
            line_root = os.path.join(point_dynamics_root, line_folder, protocol_folder)
            line_dest_folder = os.path.join(embeddings_dest_folder, line_folder, protocol_folder)
            if os.path.isfile(os.path.join(line_dest_folder, f'encoded_trajs_{traj_length}.npy')):
                pbar.update()
                continue # we do not reembed previously embedded trajectories
            embed_line(line_root,
                       line_dest_folder,
                       encoder,
                       traj_length,
                       start_time,
                       tracker,
                       trainer.len_traj,
                       trainer.n_features)
            pbar.update()


if __name__ == '__main__':
    point_dynamics_root = '/home/alexandre/workspace/larva_dataset/alzheimer_lines/t2'
    embeddings_dest_folder = '/home/alexandre/workspace/t2_embeddings'
    tracker = Tracker.T5
    start_time = Tracker.get_stimulus_time(tracker)
    traj_length = 150 # corresponds to roughly 15 seconds of post-activation activity
    model_path = '/home/alexandre/workspace/test_project/training_log/2022-01-14_11-39-00_100_'

    main(model_path, point_dynamics_root, embeddings_dest_folder, traj_length, start_time, tracker)