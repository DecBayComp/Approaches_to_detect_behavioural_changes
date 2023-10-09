import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import functools # introspection for decorators
import itertools
import torch.utils.data

from .larva_dataset import LarvaDataset

N_BATCHES = 1000
MAX_PRED = 100
LONG_TRAJ = 150
WINDOWS= {'t5':[40,60]} #, 't15':[25,45]}

def rotate(data, R):
    '''Applies the linear transformation R to a batch of data

       Parameters:
           data : tensor of shape n_batch * (2*n_trackpoints) * T , the second set of dimensions corresponds to the coordinate of the larva on the plane.
           R    : batch of linear transformation, i.e. batch * 2 * 2 tensor
    '''
    data = torch.einsum('bij,bist->bjst', R, data) # This is a simple matrix product expressed with a contraction operator.
                                                   # b : batch ; i : initial coordinates ; j : new coordinates ; s : segment ; t : time
    return data

def compute_R(data, flip_flag=False):
    '''Construct the appropriate rotation matrix to realign the data
       Uses the vector linking the tail to the head as a surrogate for body direction

       Parameters:
           data : a single sample of shape (2*n_trackpoints) * T
           flip_flag : bool, whether or not to randomly flip along y-axis
    '''
    p1 = data.mean(dim=-1)[:,:,0]  # mean head coordinates
    p2 = data.mean(dim=-1)[:,:,-1] # mean tail coordinates

    direction = p1-p2
    unit_vector = direction/torch.linalg.norm(direction, axis=-1, keepdims=True)
    cos, sin = unit_vector[:,0], unit_vector[:,1]

    if flip_flag:
        flip = np.random.random() > 0.5
        if flip:
            elems = [cos,sin,-sin,cos]
        else:
            elems = [cos,sin,sin,-cos]
    else:
        elems = [cos,sin,-sin,cos]

    return torch.stack(elems, dim=-1).reshape(-1,2,2).transpose(1,2) # do we need the extra dimension in front ? Yes we do, since each sample has a different rotation matrix

def batch_rotator(func):
    '''Retrieve a batch and rotate it. func should be the __next__ function of an iterable of batches.'''
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        batch = func(*args, **kwargs)
        R = compute_R(batch.present)
        batch.past = rotate(batch.past, R)
        batch.present = rotate(batch.present, R)
        batch.future = rotate(batch.future, R)
        return batch
    return wrapper

class CustomBatch:
    def __init__(self, data, len_pred):
        transposed_data = list(zip(*data))
        trajs = torch.stack(transposed_data[0], 0).float()
        # try:
        self.label = {}
        list_of_metadata_dicts = transposed_data[1]
        self.label['present_label'] = torch.Tensor([metadata_dict['present_label'] for metadata_dict in list_of_metadata_dicts])
        self.label['future_label'] = torch.Tensor([metadata_dict['future_label'] for metadata_dict in list_of_metadata_dicts])
        self.label['larva_path'] = [metadata_dict['larva_path'] for metadata_dict in list_of_metadata_dicts]
        self.label['start_point'] = [metadata_dict['start_point'] for metadata_dict in list_of_metadata_dicts]
        self.label['idx'] = [metadata_dict['idx'] for metadata_dict in list_of_metadata_dicts]
        # except:
            # self.label = torch.stack(transposed_data[1]).float()

        self.past = trajs[...,:len_pred]
        self.future = trajs[...,-len_pred:]
        self.present = trajs[...,len_pred:-len_pred] if len_pred else trajs
        self.device = trajs.device
        
    def pin_memory(self):
        self.label['present_label'] = self.label['present_label'].pin_memory()
        self.label['future_label'] = self.label['future_label'].pin_memory()
        self.past = self.past.pin_memory()
        self.present = self.present.pin_memory()
        self.future = self.future.pin_memory()
        return self
        
    def to(self, device):
        self.label['present_label'] = self.label['present_label'].to(device)
        self.label['future_label'] = self.label['future_label'].to(device)
        self.past = self.past.to(device)
        self.present = self.present.to(device)
        self.future = self.future.to(device)
        self.device = device
        return self


class DataRun:
    '''Representation of the dataset at runtime.

       A new sample is accessed through the method 'DataRun.sample'
    '''

    def __init__(self, n_features, len_traj, len_pred, batch_size, num_workers, device, data_dir=None, **kwargs):
        self.len_pred = len_pred
        self.len_traj = len_traj
        self.batch_size = batch_size
        self.n_features = n_features
        self.device = device
        self.data_dir = data_dir
        self.num_workers = num_workers
    
        # Train/Test split. We keep 10000 samples for validation. Generator is fixed for repeatability.
        larva_dataset = LarvaDataset(data_dir, n_features//2, len_traj, len_pred)
        n_samples = len(larva_dataset)
        n_train = int(0.8*n_samples)
        n_val = n_samples-n_train
        train_larva_dataset, val_larva_dataset = torch.utils.data.random_split(larva_dataset, [n_train, n_val],
                                                                                generator=torch.Generator().manual_seed(42))
        self._train_dataset = train_larva_dataset
        self._val_dataset   = val_larva_dataset

        # construct training dataloader
        train_dataloader = torch.utils.data.DataLoader(self._train_dataset,
                                                       batch_size=self.batch_size,
                                                       shuffle=True,
                                                       pin_memory=False,
                                                       collate_fn=self._collate_batch,
                                                       drop_last=True)
        self.train_dataloader = iter(itertools.cycle(train_dataloader))

        # construct validation dataloader
        val_dataloader = torch.utils.data.DataLoader(self._val_dataset,
                                                     batch_size=self.batch_size,
                                                     shuffle=True,
                                                     pin_memory=False,
                                                     collate_fn=self._collate_batch,
                                                     drop_last=True)
        self.n_val_batches = len(val_dataloader)
        self.val_dataloader = iter(itertools.cycle(val_dataloader))

    def sample(self, train_or_val):
        '''Return a sample either from the train dataloader or the validation dataloader

           train_or_val : string, 'train' or 'val'. Whether to sample from the training dataloader or the validation dataloader.
        '''

        if train_or_val == 'train':
            return next(self.train_dataloader).to(self.device)
        elif train_or_val == 'val':
            return next(self.val_dataloader).to(self.device)
        else:
            raise ValueError('train_or_val should be one of train, val')

    def _collate_batch(self, samples):
        return CustomBatch(samples, self.len_pred)

    def _collate_long_trajs(self, samples):
        return CustomBatch(samples, 0)