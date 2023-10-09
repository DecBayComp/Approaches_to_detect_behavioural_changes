import os
import argparse
import json
import multiprocessing

import numpy as np
import torch
import matplotlib
import pandas as pd
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
    preds = pd.read_csv(os.path.abspath(os.path.join(os.path.dirname(__file__), args.pred_file)), index_col='index')
    
    
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

        # TODO changer cette boucle pour lire les données
        for _, (t, date, line, larva_number) in larvae.iterrows():
            # construct filename
            line = line.replace('_p','/p').replace('#/p','#p').replace('_ch','/ch').replace('#/ch','#ch').replace('_LexAop','@LexAop').replace('_LexAop','@LexAop').replace('_20XUAS','@20XUAS').replace('_UAS_TNT','@UAS_TNT')
            line, protocol = line.split('/')
            purged_line = line.replace('@', '_')
            purged_protocol = protocol.split('@')[0]
            fn = '_'.join(['Point_dynamics', 't2', purged_line, purged_protocol, 'larva_id', date, 'larva_number', str(larva_number)])+'.txt'
            path = os.path.join(args.point_dynamics_root, 't2', line, protocol, date, fn)
            if os.path.isfile(path):
                to_preprocess.append((path, t, encoder.len_traj))
            else:
                print(path)

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

    model_params = torch.load(os.path.join(config['log_dir'], 'best_validated_model.pt'))
    path_saved_embedder = os.path.join(config['log_dir'], 'umap', 'parametric_umap')
    trainer = Trainer(path_saved_embedder=path_saved_embedder, **config)
    trainer.load_state_dict(model_params)
    _, eval_results = trainer.evaluate(final=True)

    ########################################################
    #
    #                 OLD vs NEW for bendy group
    #
    #########################################################

    if False:
        umapper = UMAP()

        _, axs = plt.subplots(2,3, figsize=(24,16))
        plt.subplots_adjust(wspace=0, hspace=0)
        for ax in axs.ravel():
            ax.set_xticks([])
            ax.set_yticks([])

        for row, old_idx, old_label in zip([0,1], [1,5], ['bend', 'roll']):
            print(old_label)
            axs[row, 0].set_ylabel(old_label)
            for new_idx, new_behavior in enumerate(['head_cast', 'static_bend', 'c_shape']):
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
                axs[row, new_idx].scatter(old_2d[:,0], old_2d[:,1], c=trainer.visu.colors[old_idx], s=1., marker='o')
                axs[row, new_idx].scatter(new_2d[:,0], new_2d[:,1], c='xkcd:royal blue', s=1., marker='o')
                axs[row, new_idx].legend(handles=[Patch(fc=trainer.visu.colors[old_idx], label=old_label),
                                     Patch(fc='xkcd:royal blue', label=new_behavior)],
                                     loc='upper right')

        for new_idx, new_behavior in enumerate(['head_cast', 'static_bend', 'c_shape']):
            axs[1, new_idx].set_xlabel(new_behavior)

        plt.savefig('figure_5A.jpg')

    ########################################################
    #
    #                 OLD vs NEW for head-tail
    #
    #########################################################


    if False:
        umapper = UMAP()

        _, axs = plt.subplots(1,2, figsize=(16,8))
        plt.subplots_adjust(wspace=0, hspace=0)
        for ax in axs.ravel():
            ax.set_xticks([])
            ax.set_yticks([])

        new_idx, new_behavior = 0, 'head_tail'

        for col, old_idx, old_label in zip([0,1], [3,4], ['hunch', 'back']):
            print(old_label)
            axs[col].set_xlabel(old_label)

                
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
            axs[col].scatter(old_2d[:,0], old_2d[:,1], c=trainer.visu.colors[old_idx], s=2., marker='o')
            axs[col].scatter(new_2d[:,0], new_2d[:,1], c='xkcd:pumpkin', s=2., marker='o')
            axs[col].legend(handles=[Patch(fc=trainer.visu.colors[old_idx], label=old_label),
                                     Patch(fc='xkcd:pumpkin', label=new_behavior)],
                                     loc='upper right')


        axs[0].set_ylabel(new_behavior)

        plt.savefig('figure_5B.jpg')


    ########################################################
    #
    #                NEW vs NEW for bendy group
    #
    #########################################################
    
    if False:
        umapper = UMAP()

        _, axs = plt.subplots(1,3, figsize=(24,8))
        plt.subplots_adjust(wspace=0, hspace=0)

        for ax in axs.ravel():
            ax.set_xticks([])
            ax.set_yticks([])

        colors = {'static_bend':'xkcd:neon purple',
                  'c_shape':'xkcd:tangerine',
                  'head_cast':'xkcd:neon blue'}
        from itertools import combinations
        for col, (label1, label2) in enumerate(combinations(['static_bend', 'c_shape', 'head_cast'], 2)):
            print(label1, label2)
            # continue
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
            axs[col].scatter(embeds1_2d[:,0], embeds1_2d[:,1], c=colors[label1], s=1., marker='x')
            axs[col].scatter(embeds2_2d[:,0], embeds2_2d[:,1], c=colors[label2], s=1., marker='+')
            axs[col].legend(handles=[Patch(fc=colors[label1], label=label1), Patch(fc=colors[label2], label=label2)], loc='upper right')


        plt.savefig('figure_5C.jpg')

    # Selected figures (discussion with JB 19/06/2023)
    if True:
        umapper = UMAP()


        # head-tail vs back
        back_embeds = eval_results['embeds'][eval_results['labels'] == 4]
        headtail_embeds = exotic_embeddings_dict['head_tail']
        n_embeds = min(len(back_embeds), len(headtail_embeds))
        back_embeds = back_embeds[np.random.permutation(len(back_embeds))[:n_embeds]]
        headtail_embeds = headtail_embeds[np.random.permutation(len(headtail_embeds))[:n_embeds]]
        stacked_embeds = np.vstack([back_embeds, headtail_embeds])
        embeds2d = umapper.fit_transform(stacked_embeds)

        back_2d = embeds2d[:n_embeds]
        headtail_2d = embeds2d[n_embeds:]

        plt.figure()
        plt.scatter(back_2d[:,0], back_2d[:,1], c=trainer.visu.colors[4], s=1., marker='o')
        plt.scatter(headtail_2d[:,0], headtail_2d[:,1], c='xkcd:royal blue', s=1., marker='o')
        plt.legend(handles=[Patch(fc=trainer.visu.colors[4], label='back'),
                                Patch(fc='xkcd:royal blue', label='head-tail')],
                                loc='upper right')
        plt.savefig(os.path.join(os.path.dirname(__file__), 'figure_4H.jpg'))

        # roll vs c-shape

        plt.figure()
        roll_embeds = eval_results['embeds'][eval_results['labels'] == 5]
        cshape_embeds = exotic_embeddings_dict['c_shape']
        n_embeds = min(len(roll_embeds), len(cshape_embeds))
        roll_embeds = roll_embeds[np.random.permutation(len(roll_embeds))[:n_embeds]]
        cshape_embeds = cshape_embeds[np.random.permutation(len(cshape_embeds))[:n_embeds]]
        stacked_embeds = np.vstack([roll_embeds, cshape_embeds])
        embeds2d = umapper.fit_transform(stacked_embeds)

        roll_2d = embeds2d[:n_embeds]
        cshape_2d = embeds2d[n_embeds:]
        plt.scatter(roll_2d[:,0], roll_2d[:,1], c=trainer.visu.colors[5], s=1., marker='o')
        plt.scatter(cshape_2d[:,0], cshape_2d[:,1], c='xkcd:royal blue', s=1., marker='o')
        plt.legend(handles=[Patch(fc=trainer.visu.colors[5], label='roll'),
                                Patch(fc='xkcd:royal blue', label='c-shape')],
                                loc='upper right')

        plt.savefig(os.path.join(os.path.dirname(__file__), 'figure_4I.jpg'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_file', default='DF_final_prediction_23_02_t2.csv')
    parser.add_argument('--point_dynamics_root', default='/home/alexandre/workspace/larva_dataset/point_dynamics_5')
    parser.add_argument('--model', default='/home/alexandre/workspace/maggotuba_models/maggotuba_scale_20')
    parser.add_argument('--name', default='experiment_1')
    parser.add_argument('--n_workers', default=10, type=int)
    parser.add_argument('--save', default=False, action='store_true')
    parser.add_argument('--plot', default=True, action='store_true')
    args = parser.parse_args()

    main(args)
