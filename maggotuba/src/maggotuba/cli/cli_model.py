import json
import os
import logging
import multiprocessing
import itertools


import torch
from tqdm import tqdm
import numpy as np

from maggotuba.behavior_model.models.model import Trainer
from maggotuba.behavior_model.models.neural_nets import Encoder
from maggotuba.behavior_model.data.enums import Timeslot, Feature, Tracker
import maggotuba.behavior_model.data.utils as data_utils


######################################################################################################################################
#
#                                                               TRAIN
#
######################################################################################################################################

def add_arguments_train(parser):
    parser.add_argument('--name', default=None)
    parser.add_argument('--config', default='config.json')

def train(args):
    # load config
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # create experiment name
    if args.name is None:
        experiments = os.listdir(config['log_dir'])
        experiment_ids = [e[len('experiment_'):] for e in experiments if e.startswith('experiment_')]
        experiment_ids = [int(e) for e in experiment_ids if e.isnumeric()]
        experiment_ids.append(0)
        args.name = f'experiment_{max(experiment_ids)+1}'
    config['exp_name'] = args.name
    print('experiment name : ', args.name)

    # create and populate experiment folder
    config['exp_folder'] = os.path.join(config['log_dir'], config['exp_name'])
    os.mkdir(config['exp_folder'])
    os.mkdir(os.path.join(config['exp_folder'], 'visu'))
    config['log_dir'] = config['exp_folder']
    with open(os.path.join(config['exp_folder'], 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    # print contents of the config file
    print('----------ARGUMENTS----------')
    for k, v in config.items():
        print(k, ' : ', v)
    print('-----------------------------')

    # start training
    torch.manual_seed(config['seed'])
    trainer = Trainer(**config)
    trainer.fit()

######################################################################################################################################





######################################################################################################################################
#
#                                                               EVAL
#
######################################################################################################################################

def add_arguments_eval(parser):
    parser.add_argument('--name', default=None)
    parser.add_argument('--config', default='config.json')
    parser.add_argument('--retrain_umap', action='store_true', default=False)

def eval(args):
    if args.name is None:
        raise ValueError('Please provide an experiment name')

    # get training log dir
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

    print('Number of parameters : ', sum(p.numel() for p in trainer.parameters() if p.requires_grad))

    _, eval_results = trainer.evaluate(final=True)
    trainer.save_figs(eval_results, save_to='eval', fit_embedder=args.retrain_umap, fit_pairwise_embedder=args.retrain_umap, fit_single_embedder=args.retrain_umap, final=True)
    embeds2d = trainer.visu._embed(eval_results['embeds'], fit=False)
    np.save(os.path.join(log_dir, args.name, 'visu', 'eval', 'embeds2d.npy'), embeds2d)

######################################################################################################################################





######################################################################################################################################
#
#                                                               EMBED
#
######################################################################################################################################

def add_arguments_embed(parser):
    parser.add_argument('--name', default=None)
    parser.add_argument('--config', default='config.json')
    parser.add_argument('--n_workers', type=int, default=1)

def embed(args):
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

    embeddings_folder = os.path.join(config['log_dir'], 'embeddings')
    os.makedirs(embeddings_folder, exist_ok=True)

    pool = multiprocessing.Pool(args.n_workers)
    with tqdm() as pbar:
        for tracker, line_folder, protocol_folder in iterfiles(config['raw_data_dir']):
            line_root = os.path.join(config['raw_data_dir'], tracker, line_folder, protocol_folder)
            line_dest_folder = os.path.join(embeddings_folder, tracker, line_folder, protocol_folder)
            if os.path.isfile(os.path.join(line_dest_folder, f'encoded_trajs.npy')):
                pbar.update()
                continue # we do not reembed previously embedded trajectories
            embed_line(line_root,
                       line_dest_folder,
                       encoder,
                       tracker,
                       config['len_traj'],
                       pool)
            pbar.update()


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


# define iterator over Point_dynamics files
def iterfiles(root):
    for tracker in os.listdir(root):
        if not(tracker in ['t2', 't5', 't15']):
            continue

        for line in os.listdir(os.path.join(root, tracker)):
            if not os.path.isdir(os.path.join(root, tracker, line)):
                continue
            for protocol in os.listdir(os.path.join(root, tracker, line)):
                if not os.path.isdir(os.path.join(root, tracker, line, protocol)):
                    continue
                # if not(protocol.startswith('p_8_45s1x30s0s#p_8_105s10x2s10s#n#n@100')
                #          or protocol.startswith('p_3gradient1_45s1x30s0s#p_3gradient1_105s10x2s10s#n#n@100')):
                #          continue
                yield tracker, line, protocol
    

def preprocess_file(path, tracker, window_length, avg_before_length):
    data = np.loadtxt(path).reshape((-1, len(Feature)))
    start_time = Tracker.get_stimulus_time(tracker)

    if not sum(data[:,0] >= start_time):
        logging.debug('no point after start_time')
        return None
    start_point = np.min(np.argwhere(data[:,0] >= start_time))

    w = data[start_point:start_point+window_length]
    if len(w) != window_length:
        logging.debug('not enough points')
        return None

    # preprocess the batch
    # rescale
    is_before = [Timeslot.from_timestamp(row[0], tracker) == Timeslot.BEFORE for row in data]

    before_data = data[is_before]
    if len(before_data):
        w[:,Feature.FIRST_COORD.value:Feature.LAST_COORD.value+1] /= np.mean(before_data[:,Feature.LENGTH.value])
    else:
        w[:,Feature.FIRST_COORD.value:Feature.LAST_COORD.value+1] /= avg_before_length

    present_indices = slice(0, window_length)
    # rotate
    w = data_utils.rotate(w, present_indices)

    # center
    w = data_utils.center_coordinates(w, present_indices)

    # select coordinates
    w = data_utils.select_coordinates_columns(w)
    w = data_utils.reshape(w)

    return w

@torch.no_grad()
def embed_line(line_root, line_dest_folder, encoder, tracker, window_length, pool):
    '''
    Embeds the snippet of length window_length (corresponding to the input length of the model)
    '''
    os.makedirs(line_dest_folder, exist_ok=True)
    tracker = Tracker[tracker.upper()]
    start_time = Tracker.get_stimulus_time(tracker)

    avg_before_length = compute_avg_before_length(line_root, tracker, pool)
    if avg_before_length is None:
        # there is no way to renormalize the data, unless we normalize it by 1 ?
        folder, protocol = os.path.split(line_dest_folder)
        folder, line = os.path.split(folder)
        with open(os.path.join(folder, 'rejected.txt'), 'a') as f:
            f.write('{} {}\n'.format(line, protocol))
        print("INFO : Skipping {} {} {} : no renormalization possible {}".format(tracker, line, protocol, line_root))
        return

    files = []
    for root, dirnames, filenames in os.walk(line_root):
        for fn in filenames:
            if fn.endswith('.txt') and fn.startswith('Point_dynamics_'):
                files.append(os.path.join(line_root, root, fn))
    preprocessed_files = pool.starmap(preprocess_file,
                                      zip(files,
                                          itertools.repeat(tracker),
                                          itertools.repeat(window_length),
                                          itertools.repeat(avg_before_length)))
    preprocessed_files = [ w for w in preprocessed_files if w is not None]

    if not preprocessed_files:
        # there is no way to renormalize the data, unless we normalize it by 1 ?
        folder, protocol = os.path.split(line_dest_folder)
        folder, line = os.path.split(folder)
        with open(os.path.join(folder, 'rejected.txt'), 'a') as f:
            f.write('{} {}\n'.format(line, protocol))
        print("INFO : Skipping {} {} {} : no samples can be constructed {}".format(tracker, line, protocol, line_root))
        return

    input_ = torch.from_numpy(np.stack(preprocessed_files))

    # convert to float to run through network
    input_ = input_.float().cpu()

    # compute the codes
    output_ = encoder(input_)

    # record
    encoded_trajs = output_.numpy()
    np.save(os.path.join(line_dest_folder, 'encoded_trajs.npy'), encoded_trajs)
