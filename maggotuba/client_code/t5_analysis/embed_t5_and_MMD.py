from tqdm import tqdm
import os
import itertools
import logging

import torch.multiprocessing as mp

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

    encoded_trajs = []
    for root, dirnames, filenames in os.walk(line_root):
        # print('progress')
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
                # print(data.shape)

                # transform the trajectory to a batch
                windowed_input = np.squeeze(np.lib.stride_tricks.sliding_window_view(time_selected_data, (window_length,time_selected_data.shape[1]))).copy()

                # preprocess the batch
                preprocessed_input = []
                present_indices = slice(prewindow, prewindow+traj_length)
                for i, w in enumerate(windowed_input):
                    # print(i)
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
                # print('computing R')
                R = compute_R(input_[:,:,:,present_indices])
                # print('rotating')
                input_ = rotate(input_, R)

                # convert to float to run through network
                # print('converting')
                input_ = input_.float()

                # compute the codes
                # print('encoding')
                with torch.no_grad():
                    output_ = encoder(input_)
                # print('recording')
                # record
                encoded_trajs.append(output_.numpy())
                # print('moving on')
    encoded_trajs = np.stack(encoded_trajs)
    np.save(os.path.join(line_dest_folder, 'encoded_trajs.npy'), encoded_trajs)
    # print('done')

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

    return s/n

def iterlines(point_dynamics_root):
    for line in os.listdir(point_dynamics_root):
        if not os.path.isdir(os.path.join(point_dynamics_root, line)):
            continue
        for protocol in os.listdir(os.path.join(point_dynamics_root, line)):
            if (not(os.path.isdir(os.path.join(point_dynamics_root, line, protocol)))
               or not(protocol.startswith('p_8') or protocol.startswith('p_3'))):
                continue
            yield line, protocol

def load_embedding(embeddings_dest_folder, line_folder, protocol_folder):
    filename = os.path.join(embeddings_dest_folder,
                            line_folder,
                            protocol_folder,
                            'encoded_trajs.npy')
    data = np.load(filename)
    return data.reshape(-1, data.shape[-1])

def aggregate_deciles(N, deciles):
    weights = np.array([0.05, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.05 ]).reshape(1,-1)
    N = N.reshape(-1, 1)
    weights = N*weights
    print(weights.shape, deciles.shape)
    print(weights[:2,:])
    weights = weights.flatten()
    deciles = deciles.flatten()
    argsort = np.argsort(deciles)
    deciles = deciles[argsort]
    weights = weights[argsort]
    cumweights = np.cumsum(weights)
    median_index = np.searchsorted(cumweights, 0.5*np.sum(weights))-1
    median = deciles[median_index]
    return median

def compute_mmd_between_line(embeddings_dest_folder, line1, protocol1, line2, protocol2, sigma, mmd_dest_file):
        embedding1 = load_embedding(embeddings_dest_folder, line1, protocol1)
        embedding2 = load_embedding(embeddings_dest_folder, line2, protocol2)

        mmd, var = compute_linear_estimator_of_mmd(embedding1, embedding2, sigma)
        with open(mmd_dest_file, 'a') as f:
            f.write("{}, {}, {}, {}, {}, {}\n".format(line1, protocol1,
                                                    line2, protocol2,
                                                    mmd, var))

def main(model_path, point_dynamics_root, embeddings_dest_folder, traj_length, start_time, tracker):
    model_args = make_args(train=False)[0]
    model_args['eval_saved_model'] = model_path
    model_args['log_dir'] = model_path

    trainer = load_model(model_args)
    encoder = trainer.autoencoder.encoder
    encoder.eval()
    
    # embed the lines
    with tqdm(total=1946) as pbar:
        for line_folder, protocol_folder in iterlines(point_dynamics_root):
            line_root = os.path.join(point_dynamics_root, line_folder, protocol_folder)
            line_dest_folder = os.path.join(embeddings_dest_folder, line_folder, protocol_folder)
            embed_line(line_root,
                       line_dest_folder,
                       encoder,
                       traj_length,
                       start_time,
                       tracker,
                       trainer.len_traj,
                       trainer.n_features)
            pbar.update()



    # compute the kernel size
    # compute deciles
    deciles = []
    N = []
    for line_folder, protocol_folder in iterlines(embeddings_dest_folder):
        if line_folder.startswith('FCF_'):
            embedding = load_embedding(embeddings_dest_folder, line_folder, protocol_folder)
            distmat = distance_matrix(embedding, embedding)
            m = distmat.shape[0]
            deciles.append(np.quantile(distmat[np.triu_indices(m,1,m)], np.linspace(0,1,11)))
            N.append(len(embedding))

    # aggregate deciles to compute median
    deciles = np.vstack(deciles)
    N = np.array(N)
    sigma = aggregate_deciles(N, deciles)
    print(sigma)


    # compute the mmd
    for line1, line2 in itertools.permutations(iterlines(embeddings_dest_folder), r=2):
        line1, protocol1 = line1
        line2, protocol2 = line2

        compute_mmd_between_line(embeddings_dest_folder, line1, protocol1, line2, protocol2, sigma, mmd_dest_file)

if __name__ == '__main__':
    point_dynamics_root = '/home/alexandre/workspace/larva_dataset/t5_t15_point_dynamics/t5_t15_point_dynamics_data/t5_point_dynamics/point_dynamics_data'
    embeddings_dest_folder = '/home/alexandre/workspace/t5_embeddings'
    mmd_dest_file = '/home/alexandre/workspace/t5_embeddings/mmd.csv'
    tracker = Tracker.T5
    start_time = Tracker.get_stimulus_time(tracker)
    traj_length = 20 # corresponds to roughly two seconds of post-activation activity
    model_path = '/home/alexandre/workspace/test_project/training_log/2022-01-14_11-39-00_100_'

    main(model_path, point_dynamics_root, embeddings_dest_folder, traj_length, start_time, tracker)