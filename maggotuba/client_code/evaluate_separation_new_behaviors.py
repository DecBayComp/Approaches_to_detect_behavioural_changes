import os
import argparse
import pickle
import json
import multiprocessing

import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt

from sklearn.svm import SVC, OneClassSVM
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix
from sklearn.neighbors import NearestNeighbors

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

    ###############################################################
    #
    #             Build the embeddings of standard and exotic behaviors
    #
    ###############################################################

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

     # load model
    encoder_params = torch.load(os.path.join(config['log_dir'], 'best_validated_encoder.pt'))

    encoder = Encoder(**config)
    encoder.load_state_dict(encoder_params)
    encoder.eval()
    encoder.to('cuda')

    # open exotic behavior index from Chloe's classifier
    print('Loading the data !!!')
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

        if len(larvae) > 15000:
            print(behavior, ' : ', len(larvae))
            larvae = larvae.sample(n=15000)

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

    # model_params = torch.load(os.path.join(config['log_dir'], 'best_validated_model.pt'))
    # path_saved_embedder = os.path.join(config['log_dir'], 'umap', 'parametric_umap')
    # trainer = Trainer(path_saved_embedder=path_saved_embedder, **config)
    # trainer.load_state_dict(model_params)
    # _, eval_results = trainer.evaluate(final=True)

    #################################################################
    #
    #           Check for separation of bend-like behaviors
    #
    #################################################################

    bendy_behaviors =  ['c_shape', 'head_cast', 'static_bend']
    lens = {k:len(exotic_embeddings_dict[k]) for k in bendy_behaviors}
    
    # Using SVM classifier and confusion matrix
    if False:
        svc = SVC()
        X = np.vstack([exotic_embeddings_dict[k] for k in bendy_behaviors])
        y = np.vstack([np.full((lens[k],1), i) for i, k in enumerate(bendy_behaviors)])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42, stratify=y)
        svc.fit(X_train, y_train)
        plot_confusion_matrix(svc, X_test, y_test, display_labels=bendy_behaviors)
        plt.show()

    # Using nearest neighbors and balls to estimate support and Jaccard-like distance
    if True:
        n_samples = 1000000
        rng = np.random.default_rng()
        from collections import defaultdict
        multiscale_intersection_list = defaultdict(list)
        from scipy.spatial import distance_matrix


        print("Computing bandwidth from persistence :")
        print("Smallest radius such that the point cloud has 2 connected components (left-right)")
        from gudhi import RipsComplex, plot_persistence_barcode, plot_persistence_diagram
        radii = {}
        for b in bendy_behaviors:
            distmat = distance_matrix(exotic_embeddings_dict[b], exotic_embeddings_dict[b], p=np.inf)
            rips_complex = RipsComplex(distance_matrix=distmat)
            simplex_tree = rips_complex.create_simplex_tree(max_dimension=0)
            diag = simplex_tree.persistence()
            diag = sorted(diag, key=lambda x: x[1][1], reverse=True)
            # deaths = np.array([d[1][1] for d in diag[:50]])
            # deaths[0] = deaths[1]
            # _, ax = plt.subplots(2,1)
            # ax[0].plot(deaths)
            # ax[1].plot(range(1, 50), deaths[1:]/deaths[:-1])
            # plt.show()
            print(b, ' : ', diag[2][1][1])
            radii[b] = diag[2][1][1]

        print("\n\n")


        X = np.vstack([exotic_embeddings_dict[k] for k in bendy_behaviors])
        min_vect = np.min(X, axis=0, keepdims=True)
        max_vect = np.max(X, axis=0, keepdims=True)

        kde = {k:NearestNeighbors(metric='minkowski', p=np.inf).fit(exotic_embeddings_dict[k]) for k in bendy_behaviors}
        neighbors_dist = {}
        b1, b2, b3 = bendy_behaviors
        samples = min_vect + (max_vect-min_vect)*rng.uniform(size=(n_samples,10))
        neighbors_dist[b1], _ = kde[b1].kneighbors(samples, n_neighbors=1, return_distance=True)
        neighbors_dist[b2], _ = kde[b2].kneighbors(samples, n_neighbors=1, return_distance=True)
        neighbors_dist[b3], _ = kde[b3].kneighbors(samples, n_neighbors=1, return_distance=True)

        radius = {}
        for b in bendy_behaviors:
            distmat = distance_matrix(exotic_embeddings_dict[b], exotic_embeddings_dict[b], p=np.inf)
            if True:
                distmat[distmat==0.] = np.inf
                max_distance_to_closest_neighbor = np.max(np.min(distmat, axis=0))
                radius[b] = max_distance_to_closest_neighbor
            else:
                radius[b] = np.max(np.quantile(distmat, 0.01, axis=0))



        print('Computing bandwidth from min distance to neighbor')
        for k, v in radius.items():
            print('    ', k, ' : ', v)
        if True:
            print("This is the smallest distance such that every point is connected to at least one other point.\n",
                  "This is by no means the smallest distance that connexifies the data space.")
        else:
            print("This is the max of the row-wise 0.01-quantile of the distance matrix.\n",
                   "Each ball contains at least 1 percent of the data.")
        
        print("\n\n")
        print('Using persistence based bandwidth.')
        radius = radii
        # "optimal" bandwidth
        # Implemented with KDTree, supposedly faster ?
        print('\n\n' + 50*'-' + '\n\n')
        print('"optimal" bandwidth')
        print('\n\n')

        intersection_dict = {}
        indicator = {}

        samples = min_vect + (max_vect-min_vect)*rng.uniform(size=(n_samples,10))

        for b in bendy_behaviors:
            indicator[b] = neighbors_dist[b] < radius[b]/2
            intersection_dict[(b,)] = np.sum(indicator[b])/len(samples)
        
        from itertools import combinations
        for b1, b2 in combinations(bendy_behaviors, 2):
            intersection_dict[(b1,b2)] = np.sum(np.logical_and(indicator[b1],
                                                               indicator[b2]))/len(samples)

            
        intersection_dict[(b1, b2, b3)] = np.sum(np.logical_and(indicator[b1],
                                                 np.logical_and(indicator[b2],
                                                                indicator[b3])))/len(samples)

        for k, v in intersection_dict.items():
            print(' int '.join(k), ' : ', v)
        print('\n\n')

        for b1, b2 in combinations(bendy_behaviors, 2):
            j_dist = intersection_dict[(b1,b2)]/(intersection_dict[(b1,)] + intersection_dict[(b2,)] - intersection_dict[(b1,b2)])
            print(f'Jaccard distance {b1} against {b2} : {j_dist}')
       
        # multiscale analyis
        if False:
            multiscale_intersection_list = defaultdict(list)
            bandwidths = np.logspace(0,1,10)
            for i, bandwidth in enumerate(bandwidths):
                print('\n\n' + 50*'-' + '\n\n')
                print('current bandwidth : ', bandwidth)
                print(i+1, '/', len(bandwidths))
                print('\n\n')

                indicator = {}
                
                for b in bendy_behaviors:
                    indicator[b] = neighbors_dist[b] < bandwidth
                    intersection_dict[(b,)] = np.sum(indicator[b])/len(samples)
                    multiscale_intersection_list[(b,)].append(intersection_dict[(b,)])
                
                from itertools import combinations
                for b1, b2 in combinations(bendy_behaviors, 2):
                    intersection_dict[(b1,b2)] = np.sum(np.logical_and(indicator[b1],
                                                                    indicator[b2]))/len(samples)

                    multiscale_intersection_list[(b1,b2)].append(intersection_dict[(b1,b2)])

                intersection_dict[(b1, b2, b3)] = np.sum(np.logical_and(indicator[b1],
                                                        np.logical_and(indicator[b2],
                                                                        indicator[b3])))/len(samples)
                multiscale_intersection_list[(b1, b2, b3)].append(intersection_dict[(b1, b2, b3)])

                for k, v in intersection_dict.items():
                    print(' int '.join(k), ' : ', v)
                print('\n\n')

                for b1, b2 in combinations(bendy_behaviors, 2):
                    j_dist = intersection_dict[(b1,b2)]/(intersection_dict[(b1,)] + intersection_dict[(b2,)] - intersection_dict[(b1,b2)])
                    print(f'Jaccard distance {b1} against {b2} : {j_dist}')
                    multiscale_intersection_list['jac_'+b1+'_'+b2].append(j_dist)

            _, ax = plt.subplots(1,2, sharex=True)
            l0, l1 = [], []
            for k, v in multiscale_intersection_list.items():
                if isinstance(k, tuple):
                    ax[0].loglog(bandwidths, v)
                    l0.append('m('+' int '.join(k)+')')
                else:
                    ax[1].loglog(bandwidths, v)
                    l1.append(k)
            ax[0].legend(labels=l0)
            ax[1].legend(labels=l1)
            plt.show()

    return

    






if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_file', default='/home/alexandre/Desktop/DF_final_prediction.pkl')
    parser.add_argument('--point_dynamics_root', default='/home/alexandre/workspace/larva_dataset/point_dynamics_5')
    parser.add_argument('--model', default='/home/alexandre/workspace/maggotuba_scale_20')
    parser.add_argument('--name', default='experiment_1')
    parser.add_argument('--n_workers', default=20, type=int)

    args = parser.parse_args()

    main(args)
