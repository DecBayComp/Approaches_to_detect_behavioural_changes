import argparse
import yaml
import json
import os
import torch
import torch.nn as nn
import copy
from datetime import datetime

path_config_folder = os.getcwd()+'/config/'

def default_config(project_folder, data_folder, len_traj, len_pred):
    config = {}

    # General.
    config['project_dir'] = project_folder
    config['seed'] = 100                                     # Seed for PyTorch's random number generator 
    config['exp_name'] = ''
    config['data_dir'] = ''
    config['raw_data_dir'] = data_folder
    config['log_dir'] = os.path.join(project_folder, 'training_log')
    config['exp_folder'] = None

    config['config'] = os.path.join(project_folder, 'config.json')
    config['num_workers'] = 4

    # Data.
    config['n_features'] = 10                                # number of coordinates in the time series
    config['len_traj'] = len_traj                                  # maximum length of the input trajectory
    config['len_pred'] = len_pred                                  # is related to the maximum length of the output trajectory
                                                          # maximum length of the output = 2*len_pred + len_traj

    # Model.
    config['dim_latent'] = 10
    config['activation'] = 'relu'
    config['enc_filters'] = [128, 64, 32, 32, 32,  16]  # number of filters in each layer of the conv2d and the conv1d module of the encoder
    config['dec_filters'] = [128, 64, 32, 32, 32,  16]  # number of filters in each layer of the conv2d and the conv1d module of the decoder
    config['enc_kernel'] = [(5,1), (1,config['len_traj']), (5,1), (1,config['len_traj']), (5,1), (1,config['len_traj'])]                           # filter size (segment_dim, time_dim)
    config['dec_kernel'] = [(1,config['len_traj']), (5,1), (1,config['len_traj']), (5,1), (1,config['len_traj']), (5,1)]                         # filter size (segment_dim, time_dim)

    config['bias'] = False
    config['enc_depth'] = 6
    config['dec_depth'] = 6
    config['init'] = 'kaiming'
    config['n_clusters'] = 2
    config['dim_reduc'] = 'UMAP'


    # Training.
    config['optim_iter'] = 1000                         # number of epochs
    config['pseudo_epoch'] = 100
    config['batch_size'] = 128
    config['lr'] = 0.005
    config['loss'] = 'MSE'
    config['cluster_penalty'] = None#'DEC'
    config['cluster_penalty_coef'] = 0.
    config['length_penalty_coef'] = 0.
    config['grad_clip'] = 100.0
    config['optimizer'] = 'adam'
    config['target'] = ['past','present','future']   # which segments are predicted from the time series
                                               # the way the multi scale decoder is setup at the moment, it doesn't make sense to specify only one of {past, future}
                                
    return config