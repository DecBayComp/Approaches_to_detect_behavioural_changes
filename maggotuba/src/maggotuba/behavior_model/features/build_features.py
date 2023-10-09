import torch

def length_larva(data):
    '''
    Returns the sum of distances between each adjacent points
    '''
    lengths = torch.zeros((data.shape[0], data.shape[-1])).to(data.device)
    for i in range(1, data.shape[2]):
        lengths += torch.sqrt((data[:,0,i,:] - data[:,0,i-1,:])**2 + (data[:,1,i,:] - data[:,1,i-1,:])**2)
    return lengths

def features_larva(data):
    '''
    Returns an array of concatenated features associated to a trajectory.
    The features considered are (in order) : 
    - sum of absolute lengths of larva
    - sum of absolute curvature (angle between first and third joint of the larva)
    - sum of absolute length variation
    - sum of absolute curvature variation
    - sum of absolute central point variation (speed of the central point)
    '''
    lengths = length_larva(data)
    curves = torch.atan((data[:,1,1,:]-data[:,1,0,:])/(data[:,0,1,:]-data[:,0,0,:])) - torch.atan((data[:,1,3,:]-data[:,1,2,:])/(data[:,0,3,:]-data[:,0,2,:]))
    features = torch.zeros(data.shape[0],5)
    features[:,0] = lengths.sum(-1)
    features[:,1] = torch.abs(curves).sum(-1)
    for t in range(data.shape[-1]-1): 
        features[:,2] += torch.abs(lengths[:,t+1]-lengths[:,t])
        features[:,3] += torch.abs(curves[:,t+1]-curves[:,t])
        features[:,4] += torch.sqrt((data[:,0,2,t+1] - data[:,0,2,t])**2 + (data[:,1,2,t+1] - data[:,1,2,t])**2)
    return (features - features.mean(0, keepdim=True))/features.std(0, keepdim=True)