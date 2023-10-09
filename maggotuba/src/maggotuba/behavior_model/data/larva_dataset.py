import torch
from torch.utils.data import Dataset
import numpy as np
import maggotuba.behavior_model.data.utils as utils
from maggotuba.behavior_model.data.enums import Feature
from maggotuba.behavior_model.data.enums import Label
import h5py



class LarvaDataset(Dataset):
    def __init__(self, 
                dataset_file: str,
                n_segments: int, 
                len_traj: int, 
                len_pred: int,
                **kwargs):

        # calls make_dataset and initializes self.samples as a list of tuples (path_to_sample, label)
        super(LarvaDataset, self).__init__()

        self.dataset_file = dataset_file
        self.file_handle = h5py.File(self.dataset_file, 'r')
        self.n_segments = n_segments
        self.len_traj = len_traj
        self.len_pred = len_pred
        assert(self.len_traj == self.file_handle['samples'].attrs['len_traj'])
        assert(self.len_pred == self.file_handle['samples'].attrs['len_pred'])
        self.len = None
        len(self)

    def __getitem__(self, idx):
        '''Get an item from the dataset. If indices is specified, the item's index in the dataset is indices[index].'''

        if idx >= self.len:
            raise IndexError()
        
        f = self.file_handle
        
        path = f'samples/sample_{idx}'
        larva_name = f[path].attrs['path']
        data = f[path][...]

        start_point = f[path].attrs['start_point']

        present_label = Label[f[path].attrs['behavior'].upper()].value
        future_label = self._future_label(data)

        sample = utils.select_coordinates_columns(data)
        sample = utils.reshape(sample)
        sample = torch.from_numpy(sample).float()

        label = torch.Tensor([present_label, future_label]).float()

        metadata = {'present_label':present_label, 'future_label':future_label, 'larva_path':larva_name, 'start_point':start_point, 'idx':idx}

        return sample, metadata

    def __len__(self):
        if self.len is None:
            with h5py.File(self.dataset_file, 'r') as f:
                self.len = f['samples'].attrs['n_samples']
        return self.len

    def __del__(self):
        self.file_handle.close()

    def _future_label(self, data):
        # create future label, to use in plotting the next transition probability
        future = data[-self.len_pred:, Feature.FIRST_LABEL.value:Feature.LAST_LABEL.value+1]
        unique, count = np.unique(np.argmax(future, axis=1), return_counts=True)
        future_label = unique[np.argmax(count)]
        return future_label