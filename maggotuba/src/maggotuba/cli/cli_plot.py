import os
import json
import multiprocessing
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from sklearn.neighbors import KernelDensity

from maggotuba.behavior_model.data.enums import Tracker, Timeslot, Feature
import maggotuba.behavior_model.data.utils as data_utils
from maggotuba.behavior_model.models.neural_nets import Encoder
import torch
import itertools
import tqdm

import random

plt.rcParams.update({'text.usetex':True, 'font.family':'serif'})


######################
#
#          PLOT THE DENSITY OF A LINE IN 2D LATENT SPACE
# 
# ###################### 
def add_arguments_plot_line_density(parser):
    parser.add_argument('--name', '-n', type=str, default=None)
    parser.add_argument('--tracker', '-t', type=str, default=None)
    parser.add_argument('--line', '-l', type=str, default=None)
    parser.add_argument('--protocol', '-p', type=str, default=None)

def plot_line_density(args):
    if args.name is None:
        raise ValueError('Please provide an experiment name')
    if args.tracker is None:
        raise ValueError('Please provide a tracker name')
    if args.line is None:
        raise ValueError('Please provide a line name')
    if args.protocol is None:
        raise ValueError('Please provide a protocol name')

    from umap.parametric_umap import load_ParametricUMAP

    
    # get training log
    with open('config.json', 'r') as f:
        config = json.load(f)
        log_dir = config['log_dir']
    
    # load config
    config_path = os.path.join(log_dir, args.name, 'config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)

    print('----------ARGUMENTS----------')
    for k, v in config.items():
        print('{} : {}'.format(k, v))
    print('-----------------------------')

    # load line
    line_path = os.path.join(log_dir, args.name, 'embeddings', args.tracker, args.line, args.protocol, 'encoded_trajs.npy')
    line = np.load(line_path)

    # load umapper
    umap_path = os.path.join(log_dir, args.name, 'umap', 'parametric_umap')
    umapper = load_ParametricUMAP(umap_path)
    line2D = umapper.transform(line)
    
    # load validation embedding
    evalembeds2d = np.load(os.path.join(log_dir, args.name, 'visu', 'eval', 'embeds2d.npy'))

    # construct background
    kde = KernelDensity().fit(evalembeds2d)
    xmin, ymin = np.min(evalembeds2d, axis=0) - 2.
    xmax, ymax = np.max(evalembeds2d, axis=0) + 2.
    xx, yy = np.meshgrid(np.arange(xmin, xmax, 0.1), np.arange(ymin, ymax, 0.1))
    grid = np.hstack((xx.reshape(-1,1), yy.reshape(-1,1)))
    z = np.exp(kde.score_samples(grid).reshape(xx.shape))

    # construct plot
    plt.figure(figsize=(4,4))
    plt.pcolormesh(xx, yy, z, cmap='Greys')
    plt.scatter(line2D[:,0], line2D[:,1], c='r')
    plt.axis('equal')
    plt.show()


######################
#
#          COMPARE THE DENSITIES OF TWO LINES IN THE 2D LATENT SPACE
# 
# ###################### 

from maggotuba.mmd import computeWitnessFunctionOnMesh

def add_arguments_compare_lines(parser):
    parser.add_argument('--name', '-n', type=str, default=None)
    parser.add_argument('--line1', '-l1', type=str, nargs='*', default=None, help='tracker1 line1 protocol1')
    parser.add_argument('--line2', '-l2', type=str, nargs='*', default=None, help='tracker2 line2 protocol2')
    parser.add_argument('--dest', type=str, default=None, help='tracker2 line2 protocol2')

def compare_lines(args):
    if args.name is None:
        raise ValueError('Please provide an experiment name')
    if args.line1 is None:
        raise ValueError('Please provide a reference for the first line')
    if args.line2 is None:
        raise ValueError('Please provide a reference for the second line')
    
    from umap.parametric_umap import load_ParametricUMAP


    t1, l1, p1 = args.line1
    t2, l2, p2 = args.line2

    # get training log
    with open('config.json', 'r') as f:
        config = json.load(f)
        log_dir = config['log_dir']

    # load lines
    embeds1 = np.load(os.path.join(log_dir, args.name, 'embeddings', t1, l1, p1, 'encoded_trajs.npy'))
    embeds2 = np.load(os.path.join(log_dir, args.name, 'embeddings', t2, l2, p2, 'encoded_trajs.npy'))

    # Load the umapper
    umapper = load_ParametricUMAP(os.path.join(config['log_dir'], args.name, 'umap', 'parametric_umap'))

    # compute 2d embeddings
    embeds1_2d = umapper.transform(embeds1)
    embeds2_2d = umapper.transform(embeds2)

    kernel_size = 1.0
    xmin = np.quantile(np.concatenate([embeds1_2d[:,0], embeds2_2d[:,0]]), 0.01)-1
    ymin = np.quantile(np.concatenate([embeds1_2d[:,1], embeds2_2d[:,1]]), 0.01)-1
    xmax = np.quantile(np.concatenate([embeds1_2d[:,0], embeds2_2d[:,0]]), 0.99)+1
    ymax = np.quantile(np.concatenate([embeds1_2d[:,1], embeds2_2d[:,1]]), 0.99)+1
    
    # Create mesh
    span_x = np.linspace(xmin, xmax, 200)
    span_y = np.linspace(ymin, ymax, 200)
    xx, yy = np.meshgrid(span_x, span_y)

    kde1 = gaussian_kde(embeds1_2d, xx, yy)
    kde2 = gaussian_kde(embeds2_2d, xx, yy)
    max_kde = np.max(np.concatenate((kde1, kde2)))

    wf = computeWitnessFunctionOnMesh(xx, yy, embeds1_2d, embeds2_2d, kernel_size)
    wf_m = np.max(np.abs(wf))

    fig, axs = plt.subplots(1, 3, figsize=(20,6), sharex=True, sharey=True)
    im = axs[0].imshow(kde1, origin='lower', extent=[xmin, xmax, ymin, ymax], cmap='Blues', interpolation='nearest', vmin=0.,    vmax=max_kde)
    plt.colorbar(im, ax=axs[0])
    im = axs[1].imshow(kde2, origin='lower', extent=[xmin, xmax, ymin, ymax], cmap='Reds',  interpolation='nearest', vmin=0.,    vmax=max_kde)
    plt.colorbar(im, ax=axs[1])
    im = axs[2].imshow(wf,   origin='lower', extent=[xmin, xmax, ymin, ymax], cmap='RdBu',  interpolation='nearest', vmin=-wf_m, vmax=wf_m)
    plt.colorbar(im, ax=axs[2])

    axs[0].set_title((l1+'\n'+p1).replace('#', '\\#'))
    axs[1].set_title((l2+'\n'+p2).replace('#', '\\#'))
    axs[2].set_title('Witness function')

    axs[0].set_xlim(xmin, xmax)
    axs[0].set_ylim(ymin, ymax)
    if args.dest is None:
        plt.show()
    else:
        print(50*'@'+'\n'+50*'@'+'\n',args.dest)
        plt.savefig(args.dest)

def gaussian_kde(data, xx, yy, bandwidth=1.):
    mesh_points = np.vstack([xx.flatten(), yy.flatten()])
    mesh_points = mesh_points.reshape(1, *mesh_points.shape)
    data = data.reshape(*data.shape, 1)

    N = np.sqrt(2.*np.pi*bandwidth**2)
    D2 = np.sum((mesh_points-data)**2, axis=1)

    return 1./N * np.mean(np.exp(-0.5*D2/bandwidth**2), axis=0).reshape(xx.shape)


######################
#
#          PLOT THE TRAJECTORIES IN LATENT SPACE OF ONE LINE
# 
####################### 

def add_arguments_plot_trajectories(parser):
    parser.add_argument('--name', '-n', type=str, default=None)
    parser.add_argument('--line', '-l', type=str, nargs='*', default=None, help='tracker1 line1 protocol1')
    parser.add_argument('--visu', type=str, default='umap')
    parser.add_argument('--n_workers', type=int, default=5)
    parser.add_argument('--n_trajs', type=int, default=None)

def plot_trajectories(args):
    if args.name is None:
        raise ValueError('Please provide an experiment name')
    if args.line is None:
        raise ValueError('Please provide a reference for the line')

    from umap.parametric_umap import load_ParametricUMAP



    tracker, line, protocol = args.line

    # get training log
    with open('config.json', 'r') as f:
        config = json.load(f)
    log_dir = config['log_dir']
    line_folder = os.path.join(config['raw_data_dir'], tracker, line, protocol)

    # load config
    config_path = os.path.join(log_dir, args.name, 'config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)

    print('----------ARGUMENTS----------')
    for k, v in config.items():
        print('{} : {}'.format(k, v))
    print('-----------------------------')

    # load model
    model_params = torch.load(os.path.join(config['log_dir'], 'best_validated_encoder.pt'))

    encoder = Encoder(**config)
    encoder.load_state_dict(model_params)
    encoder.eval()
    encoder.to('cpu')

    print("Compute normalization constant...")
    pool = multiprocessing.Pool(args.n_workers)
    avg_before_length = compute_avg_before_length(line_folder, Tracker[tracker.upper()], pool)
    pool.close()
    pool.terminate()

    print("Embed the trajectories...")
    trajs = []

    paths = []
    for date in os.listdir(line_folder):
        paths += [os.path.join(line_folder, date, fn) for fn in os.listdir(os.path.join(line_folder, date))]

    if args.n_trajs is not None:
        paths = random.sample(paths, args.n_trajs)

    for path in tqdm.tqdm(paths):
        trajs.append(embed_file(path, encoder, avg_before_length, Tracker[tracker.upper()]))

    if args.visu == 'umap':
        umap_path = os.path.join(log_dir, args.name, 'umap', 'parametric_umap')
        umapper = load_ParametricUMAP(umap_path)

        for idx, (traj, _) in tqdm.tqdm(enumerate(trajs)):
            trajs[idx] = umapper.transform(traj)

        # display training dataset in the background
        evalembeds2d = np.load(os.path.join(log_dir, args.name, 'visu', 'eval', 'embeds2d.npy'))
        kde = KernelDensity().fit(evalembeds2d)
        xmin, ymin = np.min(evalembeds2d, axis=0) - 2.
        xmax, ymax = np.max(evalembeds2d, axis=0) + 2.
        xx, yy = np.meshgrid(np.arange(xmin, xmax, 0.1), np.arange(ymin, ymax, 0.1))
        grid = np.hstack((xx.reshape(-1,1), yy.reshape(-1,1)))
        z = np.exp(kde.score_samples(grid).reshape(xx.shape))

        # construct plot
        plt.figure(figsize=(4,4))
        plt.pcolormesh(xx, yy, z, cmap='Greys')
        for traj in trajs:
            plt.plot(traj[:,0], traj[:,1], c='r', alpha=1./(len(trajs)**(.25)))
        plt.axis('equal')
        plt.title(f'{tracker} {line} {protocol}')
        plt.show()

    elif args.visu == 'first2':

        # construct plot
        plt.figure(figsize=(4,4))
        for traj, _ in trajs:
            plt.plot(traj[:,0], traj[:,1], c='r', alpha=1./(len(trajs)**(.25)))
        plt.axis('equal')
        plt.title(f'{tracker} {line} {protocol}')

        plt.show()

    elif args.visu == 'ts10':

        fig, axs = plt.subplots(10, 1, sharex=True, sharey=True)
        plt.subplots_adjust(hspace=0.)
        for traj, t in trajs:
            for dim, ax in enumerate(axs):
                ax.plot(t, traj[:,dim])
        fig.suptitle(f'{tracker} {line} {protocol}')
        plt.show()

    else:
        try:
            n_dims = int(args.visu)
        except:
            raise ValueError('Bad value for visu')

        nrows = int(np.sqrt((n_dims-1)*(n_dims-2)//2))+1
        
        # construct plot
        fig, axs = plt.subplots(nrows, nrows)
        for (dim1, dim2), ax in zip(itertools.combinations(range(n_dims), r=2), axs.ravel()):
            for traj, _ in trajs:
                ax.plot(traj[:,dim1], traj[:,dim2], c='r', alpha=1./(len(trajs)**(.25)))
            ax.axis('equal')
            ax.set_title(f'{dim1+1}th dim against {dim2+1}th dim')
        
        fig.suptitle(f'{tracker} {line} {protocol}')
        plt.show()

def compute_avg_before_length(line_root, tracker, pool):
    s = 0
    n = 0

    files = []
    for root, dirnames, filenames in os.walk(line_root):
        for fn in filenames:
            if fn.endswith('.txt') and fn.startswith('Point_dynamics_'):
                files.append(os.path.join(line_root, root, fn))
    length = pool.starmap(compute_before_length, zip(files, itertools.repeat(tracker)))
    length = [l for l in length if l is not None]
    return np.mean(length) if len(length) else None

def compute_before_length(filename, tracker):
    data = np.loadtxt(filename).reshape( (-1,len(Feature)) )
    is_before = [Timeslot.from_timestamp(row[0], tracker) == Timeslot.BEFORE for row in data]
    after_setup_before_activation_data = data[is_before]
    return np.mean(after_setup_before_activation_data[:, Feature.LENGTH.value]) if len(after_setup_before_activation_data) else None

@torch.no_grad()
def embed_file(filename, encoder, avg_before_length, tracker):
    data = np.loadtxt(filename)
    len_traj = encoder.len_traj
    T = len(data)

    # rescale
    is_before = [Timeslot.from_timestamp(row[0], tracker) == Timeslot.BEFORE for row in data]
    before_data = data[is_before]

    if len(before_data):
        data[:,Feature.FIRST_COORD.value:Feature.LAST_COORD.value+1] /= np.mean(before_data[:,Feature.LENGTH.value])
    else:
        data[:,Feature.FIRST_COORD.value:Feature.LAST_COORD.value+1] /= avg_before_length

    present_indices = slice(0, len_traj)
    windowed_data = []
    for t in range(T-len_traj):
        w = data[t:t+len_traj].copy()
        # rotate
        w = data_utils.rotate(w, present_indices)

        # center
        w = data_utils.center_coordinates(w, present_indices)

        # select coordinates
        w = data_utils.select_coordinates_columns(w)
        w = data_utils.reshape(w)
        windowed_data.append(w)

    input_ = torch.from_numpy(np.stack(windowed_data)).float()
    output_ = encoder(input_).numpy()

    return output_, data[:-len_traj,0]
