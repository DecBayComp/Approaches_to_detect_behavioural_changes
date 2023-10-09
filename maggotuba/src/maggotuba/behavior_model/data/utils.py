import numpy as np
from .enums import Feature



def rescale_coordinates(sample, indices):
    len_larva = np.mean(sample[indices,Feature.LENGTH.value])
    sample[:,Feature.FIRST_COORD.value:] = sample[:,Feature.FIRST_COORD.value:] / len_larva
    return sample

def select_coordinates_columns(sample):
    return sample[:,Feature.FIRST_COORD.value:]

def reshape(sample):
    sample = sample.transpose()                       # feature * time
    sample = sample.reshape(-1, 2, sample.shape[-1])  # {segment from head to tail} * {'x', 'y'} * time
    sample = sample.transpose(1,0,2)                    # {'x', 'y'} * {segment from head to tail} * time
    return sample

def compute_rotation_matrix(data):
    '''Construct the appropriate rotation matrix to realign the data
       Uses the vector linking the tail to the head as a surrogate for body direction

       Parameters:
           data : a single sample of shape T * (2*n_trackpoints) 
    '''
    mean_coordinates = np.mean(data, axis=0)

    p1 = mean_coordinates[:2]  # head
    p2 = mean_coordinates[-2:] # tail

    direction = p1-p2
    unit_vector = direction/np.linalg.norm(direction)
    cos, sin = unit_vector
    matrix = np.array([[cos,sin],
                      [-sin,cos]])
    return matrix

def rotate(sample, indices):
    coords = sample[:,Feature.FIRST_COORD.value:Feature.LAST_COORD.value+1].copy()
    matrix = compute_rotation_matrix(coords[indices])
    coords = np.stack([coords[:,::2], coords[:,1::2]], axis=-1)
    coords = np.einsum('ji,tpi->tpj', matrix, coords)
    coords = coords.reshape(coords.shape[0],-1)
    sample[:,Feature.FIRST_COORD.value:Feature.LAST_COORD.value+1] = coords
    return sample

def center_coordinates(sample, indices):
    n_segments = (Feature.LAST_COORD.value - Feature.FIRST_COORD.value + 1)//2
    bias_x = np.mean(sample[indices,Feature.X_MID_SEGMENT.value])
    bias_y = np.mean(sample[indices,Feature.Y_MID_SEGMENT.value])
    sample[:,Feature.FIRST_COORD.value:] = sample[:,Feature.FIRST_COORD.value:] - np.tile([bias_x,bias_y], n_segments).reshape(1,-1)
    return sample