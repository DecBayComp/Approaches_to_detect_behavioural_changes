'''
Script to load a model and perform a bunch of cluster analysis on its latent space
'''


import itertools
import os
import json

import matplotlib.pyplot as plt
import matplotlib
import matplotlib.animation as animation
from matplotlib.backend_bases import MouseButton

import torch
import numpy as np

import gudhi
from gudhi.clustering.tomato import Tomato

from sklearn.metrics import silhouette_score

from scipy.spatial import distance_matrix

from maggotuba.behavior_model.models.model import Trainer

from tqdm import tqdm, trange
import h5py
import pickle
import multiprocessing as mp

from maggotuba.clustering_app.app import App

# matplotlib.use('Qt5Agg')


clist = ['#17202A', '#C0392B', '#8BC34A', '#2E86C1', '#26C6DA', '#F1C40F']

def load_model(args):
    if args.name is None:
        raise ValueError('Please provide an experiment name')

    # get training log
    with open(args.config, 'r') as f:
        config = json.load(f)
        log_dir = config['log_dir']
    
    # load config
    config_path = os.path.join(log_dir, args.name, args.config)
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    torch.manual_seed(config['seed'])


    print('----------ARGUMENTS----------')
    for k, v in config.items():
        print('{} : {}'.format(k, v))
    print('-----------------------------')

    model_params = torch.load(os.path.join(config['log_dir'], 'best_validated_model.pt'))

    path_saved_embedder = os.path.join(config['log_dir'], 'umap', 'parametric_umap')

    trainer = Trainer(path_saved_embedder=path_saved_embedder, **config)
    trainer.load_state_dict(model_params)

    return trainer


#################################
# 
# OUTLINE FUNCTIONALITY
# 
#############################

def gather_outlines(ids, len_traj, len_pred):
    outlines = []
    filenames = [s for s in os.listdir(os.getcwd()) if s.endswith('_outlines.hdf5')]
    assert(len(filenames) == 1)
    outline_db = h5py.File(filenames[0], 'r')
    for id in ids:
        try:
            specs_x = outline_db[f'outlines/outline_{id}/fourier_x'][:]
            specs_y = outline_db[f'outlines/outline_{id}/fourier_y'][:]
        except:
            specs_x = np.full((2*len_pred+len_traj,13), np.nan, dtype=complex)
            specs_y = np.full((2*len_pred+len_traj,13), np.nan, dtype=complex)

        outlines.append({'fourier_x':specs_x, 'fourier_y':specs_y})
    return outlines



#############################


@torch.no_grad()
def main(args):
    # get training log
    with open(args.config, 'r') as f:
        config = json.load(f)
        log_dir = config['log_dir']
        len_traj = config['len_traj']
        len_pred = config['len_pred']
    
    # load config
    experiment_folder = os.path.join(log_dir, args.name)

    outlines_flag = args.outlines
    spines_flag = args.spines if outlines_flag else True

    title = os.path.basename(os.getcwd())

    # Asssemble a database
    if os.path.exists(os.path.join(experiment_folder, 'clustering.cache')) and not(args.clear_cache):
        print('Loading cached data...')
        with open(os.path.join(experiment_folder, 'clustering.cache'), 'rb') as f:
            cache = pickle.load(f)
        samples            = cache['samples']
        labels             = cache['labels']
        larva_paths        = cache['larva_paths']
        larva_start_points = cache['larva_start_points']
        embeddings         = cache['embeddings']
        embeddings2d       = cache['embeddings2d']
        outlines           = cache['outlines']

    else:
        trainer = load_model(args)
        samples = []
        labels = []
        larva_paths = []
        larva_start_points = []
        embeddings = []
        sample_ids = []
        for _ in trange(250):
            batch = trainer.data.sample('train')
            data = torch.cat([batch.past, batch.present, batch.future], dim=-1)
            samples.append(data)
            embeddings.append(trainer.autoencoder.encoder(batch.present))
            labels.append(batch.label['present_label'])
            larva_paths.append(batch.label['larva_path'])
            larva_start_points.append(batch.label['start_point'])
            sample_ids.append(batch.label['idx'])

        samples = torch.cat(samples).cpu().numpy()
        labels = torch.cat(labels).cpu().numpy()
        larva_paths = np.array(sum(larva_paths, []), dtype=object)
        larva_start_points = np.array(sum(larva_start_points, []), dtype=object)
        sample_ids = sum(sample_ids, [])
        embeddings = torch.cat(embeddings).cpu().numpy()
        embeddings2d = trainer.visu._embed(embeddings, fit=False)
        del(trainer)
        outlines = gather_outlines(sample_ids, len_traj, len_pred)
        cache = {}
        cache['samples']            = samples
        cache['labels']             = labels
        cache['larva_paths']        = larva_paths
        cache['larva_start_points'] = larva_start_points
        cache['embeddings']         = embeddings
        cache['embeddings2d']       = embeddings2d
        cache['outlines']           = outlines
        with open(os.path.join(experiment_folder, 'clustering.cache'), 'wb') as f:
            pickle.dump(cache, f)


    # Done

    app = App(title, len_traj, len_pred, embeddings, embeddings2d, labels, samples, args.depth, outlineData=outlines, plotPastFut=True, plotOutlines=outlines_flag, plotSpines=spines_flag)
    app.start()

def add_arguments(parser):
    parser.add_argument('--name', default=None)
    parser.add_argument('--config', default='config.json')
    parser.add_argument('--depth', default=None, type=int)
    parser.add_argument('--clear_cache', default=False, action='store_true')
    parser.add_argument('--outlines', default=False, action='store_true')
    parser.add_argument('--spines', default=False, action='store_true')