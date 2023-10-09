import argparse
import copy
import glob
import os
import pickle
from random import *
import torch
import torch.nn as nn
from Base_function_array import *
from Fourier import *
from Energy_loss import *
import sys 


files = sys.argv[1]
main(files)



class MODEL(nn.Module):

    def __init__(self, Cont_surface):
        super(MODEL, self).__init__()
        self.para = nn.Parameter(torch.tensor(
            Cont_surface, requires_grad=True))

def exp_lr_scheduler(optimizer, epoch, lr_decay, lr_decay_epoch):
    """Decay learning rate by a factor of lr_decay every lr_decay_epoch epochs"""
    if epoch % lr_decay_epoch:
        return optimizer
    for param_group in optimizer.param_groups:
        param_group['lr'] *= lr_decay
    return optimizer

class MATRICE():

    def __init__(self, Line, date, larva):
        self.Line = Line
        self.date = date
        self.larva = larva
        self.path_ = '/pasteur/zeus/projets/p02/hecatonchire/screens/t15/'
        self.run()

    def run(self):

        name_file = self.path_ + self.Line + '/' + self.date +'/trx.mat'
        f = h5py.File(name_file, 'r')
        trx = f.get('trx')
        Data_fourier,Time = give_data_fourier(trx,f,self.larva)

        Index_ini = Time[0]
        Index_end = Time[-1]
        learning_rate = 0.01

        Cont = np.vstack(copy.deepcopy(Data_fourier['Contour'].values)).reshape((len(Time),50,2))[Index_ini:Index_end]
        Cont = Cont
        beta = np.zeros((len(Cont),len(Cont[0])))
        beta = beta + 2
        for time in range(len(Cont)):
            for i in range(0, len(Cont[0])):
                P = Cont[time,i]
                M1 = Cont[time,i-1]
                M2 = Cont[time,(i+1)%len(Cont[0])]
                Theta = np.arctan2(M2[1]-P[1],M2[0]-P[0])
                Beta = np.arctan2(M1[1]-P[1],M1[0]-P[0])
                angle_ = np.pi - abs(Theta-Beta)
                Curvature = 2*np.tan(angle_/2)

                if Curvature > 2*np.tan((np.pi/3)/2):
                    beta[time,i] = beta[time,i]/10
                    beta[time,(i+1)%len(Cont[0])] = beta[time,i]/5
                    beta[time,i-1] = beta[time,i]/5

        beta =torch.tensor(copy.deepcopy(beta))
        target = torch.tensor(copy.deepcopy(Cont), requires_grad=True)
        features_cont = torch.tensor(copy.deepcopy(Cont))
        NewContour = copy.deepcopy(Cont)
        net = MODEL(Cont)
        optimizer = optim.SGD(net.parameters(), lr=learning_rate)
        L = 0
        LOSS_ = []

        for t in range(100000):
            optimizer.zero_grad()
            loss = Loss_function_times(net.para, features_cont,beta)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.para, 10, norm_type=2)
            net.para.grad = torch.nan_to_num(net.para.grad)
            optimizer.step()
            L = loss.detach()
            LOSS_.append(L)
            optimizer = exp_lr_scheduler(optimizer, t, 0.5, 1000)

        NewContour = net.para.detach()
        fichier_w = open('output/' + self.Line.replace('/','_') + '_date_' + self.date +'_numero_of_larva_'+str(self.larva)+'_1e_2.pkl', 'wb')
        pickle.dump((NewContour, LOSS_), fichier_w)
        fichier_w.close()

def main(arg_str):

    version = 20180409
    name_all = arg_str
    Line = name_all.split('/')[-4]+'/'+name_all.split('/')[-3]
    date = name_all.split('/')[-2]
    numero_larva = int(name_all.split('number_')[-1][:-4])
    noeuds = MATRICE(Line, date,numero_larva)
