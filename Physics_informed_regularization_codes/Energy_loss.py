import numpy as np
import copy
from Base_function_array import *
import torch
from torch import nn
from torch import optim
import torch.nn as nn
import glob
import pickle
import pandas as pd
import os
from os import path
from random import *
import math
import h5py
import torch.nn.functional as F

def Loss_function_times(target, features_cont,beta):
    mul = [2,5,2,0.5]
    F = 0
    F += torch.sum(beta*torch.sum((features_cont-target)**2,axis=2))
    F += torch.sum(torch.tanh(mul[3]*(target[0:-1] - target[1:])**2))
    F += torch.sum(mul[1] *(target[:,list(range(len(target[0])))[1:]+[0], :] - target[:,:, :])**2)

    P = target[:,list(range(len(target[0])))[1:]+[0]]
    M1 = target[:,:]
    M2 = target[:,list(range(len(target[0])))[2:]+[0]+[1]]

    Theta = torch.atan2(M2[:,:,1]-P[:,:,1],M2[:,:,0]-P[:,:,0])
    Beta = torch.atan2(M1[:,:,1]-P[:,:,1],M1[:,:,0]-P[:,:,0])
    angle_ = np.pi - abs(Theta-Beta)
    Curvature = 1*torch.tan(angle_/2)

    F += torch.sum(2*(Curvature)**2)


    return F
