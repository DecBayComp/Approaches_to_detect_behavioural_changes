import numpy as np
import pandas as pd
from Base_function_array import *

def give_fourier_coeff(Contour,G):

    n_fourier = 7;
    n_contour = len(Contour[:,0][~(np.isnan(Contour[:,0]))]);
    coeffs        = np.zeros((n_fourier, 4))

    contour_x_loc       = [Contour[i,0]-G[0] for i in range(0,n_contour)]
    contour_y_loc       = [Contour[i,1]-G[1] for i in range(0,n_contour)]
    n_contour_loc       = len(contour_x_loc);

    for k in range(0,n_fourier):
        for l in range(1,n_contour_loc):
            coeffs[k,0] += (contour_x_loc[l]) * np.cos( (k) * np.pi * (l) * 2.0 / (n_contour_loc));
            coeffs[k,1] += (contour_x_loc[l]) * np.sin( (k) * np.pi * (l) * 2.0 / (n_contour_loc));
            coeffs[k,2] += (contour_y_loc[l]) * np.cos( (k) * np.pi * (l) * 2.0 / (n_contour_loc));
            coeffs[k,3] += (contour_y_loc[l]) * np.sin( (k) * np.pi * (l) * 2.0 / (n_contour_loc));

        for l in range(0,4):
            coeffs[k,l]   = coeffs[k,l]*2.0/n_contour_loc;
    coeffs[0,0]   = coeffs[0,0]/2.0 ;
    coeffs[0,2]   = coeffs[0,2]/2.0 ;

    return coeffs


def give_fourier_contour(Contour,G):

    coeffs = give_fourier_coeff(Contour,G)

    n_reconstruct = 50;
    n_fourier     = 7;
    theta_step    = 2*np.pi/n_reconstruct;
    theta_i       = -np.pi

    contour_reconstruct = np.zeros((n_reconstruct,2))

    for k in range(0, n_fourier):
        theta = theta_i
        for l in range(0,n_reconstruct):
            contour_reconstruct[l] =[contour_reconstruct[l,0] + coeffs[k,0]*np.cos((k)*theta) + \
            coeffs[k,1]*np.sin((k)*theta),contour_reconstruct[l,1] + coeffs[k,2]*np.cos((k)*theta) + coeffs[k,3]*np.sin((k)*theta)]
            theta = theta + theta_step ;
    for l in range(0,n_reconstruct):
        contour_reconstruct[l]=contour_reconstruct[l]+G

    return contour_reconstruct




def give_data_fourier(trx,f,larva):

    for numero_larva in range(len(trx['numero_larva_num'][0])):
        if f[trx['numero_larva_num'][0][numero_larva]][0][0] == larva:
            break

    Col = ['G','Head','Tail','Contour']
    nb_larva = numero_larva

    Time = f[trx['t'][0][nb_larva]][0]
    Contour_recording = np.zeros((500, 2))
    Data_fourier = pd.DataFrame(index = Time,columns = Col)

    for n_time in range(len(Time)):

        Contour_recording[:,0] = f[trx['x_contour'][0][nb_larva]][:,n_time]
        Contour_recording[:,1] = f[trx['y_contour'][0][nb_larva]][:,n_time]
        G = [f[trx['x_center'][0][nb_larva]][0][n_time],f[trx['y_center'][0][nb_larva]][0][n_time]]
        H = [f[trx['x_head'][0][nb_larva]][0][n_time],f[trx['y_head'][0][nb_larva]][0][n_time]]
        T = [f[trx['x_tail'][0][nb_larva]][0][n_time],f[trx['y_tail'][0][nb_larva]][0][n_time]]

        Contour_reconstruct = give_fourier_contour(Contour_recording, G)

        n_head = [norm(H, Contour_reconstruct[i]) for i in range(0, len(Contour_reconstruct))].index(
            min([norm(H, Contour_reconstruct[i]) for i in range(0, len(Contour_reconstruct))]))
        n_tail = [norm(T, Contour_reconstruct[i]) for i in range(0, len(Contour_reconstruct))].index(
            min([norm(T, Contour_reconstruct[i]) for i in range(0, len(Contour_reconstruct))]))

        head = Contour_reconstruct[n_head]
        tail = Contour_reconstruct[n_tail]

        Data_fourier.loc[Time[n_time],'Head'] = head
        Data_fourier.loc[Time[n_time],'Tail'] = tail
        Data_fourier.loc[Time[n_time],'G'] = G
        Data_fourier.loc[Time[n_time],'Contour'] = Contour_reconstruct

    return Data_fourier, Time
