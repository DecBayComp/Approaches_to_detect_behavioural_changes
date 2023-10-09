import os
import argparse
import pickle
import json
import multiprocessing

import pandas as pd
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from umap import UMAP

from maggotuba.behavior_model.models.neural_nets import Encoder
from maggotuba.behavior_model.models.model import Trainer
from maggotuba.behavior_model.data.utils import rotate, center_coordinates, select_coordinates_columns, reshape
from maggotuba.behavior_model.data.enums import Feature, Tracker, Timeslot

def preprocess_file(args):
    path, t, len_traj = args
    # open file and locate the point closest to the center point
    data = np.loadtxt(path)
    center_point = np.argmin(np.abs(data[:,0]-t))
    # extract window
    minus, plus = len_traj//2, len_traj-len_traj//2
    w = data[center_point-minus:center_point+plus]
    if len(w) != len_traj:
        # print('too close to edge of file')
        return None

    # compute before length
    is_before = [Timeslot.from_timestamp(row[0], Tracker.T2) == Timeslot.BEFORE for row in data]
    after_setup_before_activation_data = data[is_before]
    if len(after_setup_before_activation_data):
        before_length = np.mean(after_setup_before_activation_data[:, Feature.LENGTH.value]) 
    else:
        # print('unrenormalizable')
        return None

    # rescale
    w[:, Feature.FIRST_COORD.value:Feature.LAST_COORD.value+1] = w[:, Feature.FIRST_COORD.value:Feature.LAST_COORD.value+1]/before_length
    # rotate
    present_indices = slice(0, len_traj)
    w = rotate(w, present_indices)
    # center
    w = center_coordinates(w, present_indices)

    # select_coordinates
    w = select_coordinates_columns(w)
    w = reshape(w)

    return w

@torch.no_grad()
def main(args):
    if args.model is None:
        print('Please provide a valid model folder path')
        exit()

    # get training log
    with open(os.path.join(args.model, 'config.json'), 'r') as f:
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
    
     # load model
    encoder_params = torch.load(os.path.join(config['log_dir'], 'best_validated_encoder.pt'))

    encoder = Encoder(**config)
    encoder.load_state_dict(encoder_params)
    encoder.eval()
    encoder.to('cuda')

    # open exotic behavior index from Chloe's classifier
    fn = os.path.abspath(args.pred_file)
    with open(fn, 'rb') as f:
        preds = pickle.load(f)
    
    # compute the embeddings
    pool = multiprocessing.Pool(args.n_workers)
    exotic_embeddings_dict = {}
    behaviors = []
    for behavior in preds['prediction'].unique():
        print(behavior.replace('\n', ''))
        larvae = preds[(preds['prediction'] == behavior) 
                     & (preds['probabilities'] == 1.0)].loc[:, 'time':'Larva']
        larvae = larvae.astype({'Larva': int}, errors='raise')

        if len(larvae) > 5000:
            larvae = larvae.sample(n=5000)

        # recover center point of the behavior bout
        center_point  = []
        for s in larvae['time']:
            s = s[1:-1]
            s1, s2 = s.split(',')
            s1, s2 = s1.strip(), s2.strip()
            t1, t2 = float(s1), float(s2)
            center_point.append(np.mean((t1,t2)))
        larvae['time'] = center_point
        
        to_embed = []
        to_preprocess = []
        for _, (t, date, line, larva_number) in larvae.iterrows():
            # construct filename
            line, protocol = line.split('/')
            purged_line = line.replace('@', '_')
            purged_protocol = protocol.split('@')[0]
            fn = '_'.join(['Point_dynamics', 't2', purged_line, purged_protocol, 'larva_id', date, 'larva_number', str(larva_number)])+'.txt'
            path = os.path.join(args.point_dynamics_root, 't2', line, protocol, date, fn)
            assert os.path.isfile(path)

            to_preprocess.append((path, t, encoder.len_traj))

        to_embed = pool.map(preprocess_file, to_preprocess)
        to_embed = [w for w in to_embed if w is not None]
        print(len(to_embed), '/', len(larvae))
        to_embed = torch.from_numpy(np.stack(to_embed)).float().to('cuda')
        embeddings = encoder(to_embed)

        embeddings = embeddings.cpu().numpy()
        behavior = behavior.replace('&', '_').replace('\n', '_').replace(' ', '_')
        behavior = behavior.replace('__', '_').replace('-', '_').lower()
        behaviors.append(behavior)

        exotic_embeddings_dict[behavior] = embeddings
        print(50*'-')

    del(encoder)
    pool.terminate()

    if args.save:
        np.save(f'{behavior}_embeddings.npy', embeddings)

    if not args.plot:
        return

    model_params = torch.load(os.path.join(config['log_dir'], 'best_validated_model.pt'))
    path_saved_embedder = os.path.join(config['log_dir'], 'umap', 'parametric_umap')
    trainer = Trainer(path_saved_embedder=path_saved_embedder, **config)
    trainer.load_state_dict(model_params)
    _, eval_results = trainer.evaluate(final=True)

    if bool('Density') == False:
        embeds = eval_results['embeds']
        embeds2d = trainer.visu._embed(embeds, fit=False)

        _, axs = plt.subplots(2,6, figsize=(24, 12), sharex=True, sharey=True)

        for i, (ax, c, lbl) in enumerate(zip(axs.ravel()[:6], trainer.visu.colors, trainer.visu.labels)):
            subset_embeds2d = embeds2d[eval_results['labels'] == i]
            cmap = type(trainer.visu).get_cmap_from_color(i)
            ax.hexbin(subset_embeds2d[:,0], subset_embeds2d[:,1], gridsize=15, cmap=cmap)
            ax.set_title(lbl)

        for ax, behavior in zip(axs.ravel()[6:12], behaviors):
            embeds = exotic_embeddings_dict[behavior]
            embeds2d = trainer.visu._embed(embeds, fit=False)
            ax.hexbin(embeds2d[:,0], embeds2d[:,1], gridsize=15, cmap='Greys')
            ax.set_title(behavior)
        plt.show()

    if bool('Pairwise umap old new') == False:
        umapper = UMAP()

        _, axs = plt.subplots(5,6, figsize=(15,15))
        plt.tight_layout()
        plt.subplots_adjust(wspace=0, hspace=0)
        for ax in axs.ravel():
            ax.set_xticks([])
            ax.set_yticks([])

        for old_idx, old_label in enumerate(trainer.visu.labels):
            print(old_label)
            axs[0, old_idx].set_title(old_label)
            for new_idx, new_behavior in enumerate(behaviors):
                # balance embeddings
                print('    ', new_behavior)
                old_embeds = eval_results['embeds'][eval_results['labels'] == old_idx]
                new_embeds = exotic_embeddings_dict[new_behavior]
                n_embeds = min(len(old_embeds), len(new_embeds))
                old_embeds = old_embeds[np.random.permutation(len(old_embeds))[:n_embeds]]
                new_embeds = new_embeds[np.random.permutation(len(new_embeds))[:n_embeds]]
                stacked_embeds = np.vstack([old_embeds, new_embeds])
                embeds2d = umapper.fit_transform(stacked_embeds)

                old_2d = embeds2d[:n_embeds]
                new_2d = embeds2d[n_embeds:]
                axs[new_idx, old_idx].scatter(old_2d[:,0], old_2d[:,1], c=trainer.visu.colors[old_idx], s=1., marker='x')
                axs[new_idx, old_idx].scatter(new_2d[:,0], new_2d[:,1], c='xkcd:pastel pink', s=1., marker='+')

        for new_idx, new_behavior in enumerate(behaviors):
            axs[new_idx, 0].set_ylabel(new_behavior)

        plt.show()


    if bool('Pairwise umap new') == True:
        umapper = UMAP()

        _, axs = plt.subplots(4,4, figsize=(15,15))
        plt.tight_layout()
        plt.subplots_adjust(wspace=0, hspace=0)

        for ax in axs.ravel():
            ax.set_xticks([])
            ax.set_yticks([])
        for i in range(1,4):
            for j in range(i):
                axs[i,j].axis('off')

        for id1, label1 in enumerate(behaviors[:-1]):
            print(label1)
            for id2, label2 in zip(range(id1+1,5), behaviors[id1+1:]):
                # balance embeddings
                print('    ', label2)
                embeds1 = exotic_embeddings_dict[label1]
                embeds2 = exotic_embeddings_dict[label2]
                n_embeds = min(len(embeds1), len(embeds2))
                embeds1 = embeds1[np.random.permutation(len(embeds1))[:n_embeds]]
                embeds2 = embeds2[np.random.permutation(len(embeds2))[:n_embeds]]
                stacked_embeds = np.vstack([embeds1, embeds2])
                embeds2d = umapper.fit_transform(stacked_embeds)

                embeds1_2d = embeds2d[:n_embeds]
                embeds2_2d = embeds2d[n_embeds:]
                axs[id1, id2-1].scatter(embeds1_2d[:,0], embeds1_2d[:,1], c='xkcd:pastel green', s=1., marker='x')
                axs[id1, id2-1].scatter(embeds2_2d[:,0], embeds2_2d[:,1], c='xkcd:pastel pink', s=1., marker='+')
                axs[id1, id2-1].legend(handles=[Patch(fc='xkcd:pastel green', label=label1), Patch(fc='xkcd:pastel pink', label=label2)])


        for new_idx, new_behavior in enumerate(behaviors[:-1]):
            axs[new_idx, new_idx].set_ylabel(new_behavior)

        for new_idx, new_behavior in enumerate(behaviors[1:]):
            axs[0, new_idx].set_title(new_behavior)

        plt.show()







    

    



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('pred_file')
    parser.add_argument('point_dynamics_root')
    parser.add_argument('--model', default=None)
    parser.add_argument('--name', default='experiment_1')
    parser.add_argument('--n_workers', default=10, type=int)
    parser.add_argument('--save', default=False, action='store_true')
    parser.add_argument('--plot', default=False, action='store_true')
    args = parser.parse_args()

    main(args)
