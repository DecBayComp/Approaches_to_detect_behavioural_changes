from torch.nn import Parameter
import torch
import numpy as np
from sklearn.manifold import MDS
from tqdm import trange

'''
Author : Hippolyte Verdier, 2021
Modified by Alexandre Blanc, 2022
'''


class MDS_prob:
    def __init__(self, target_dim=2):
        self.d = target_dim

    def fit_transform(self, D, D_sigma2, lr=0.01, n_steps=int(1e4), device='cuda'):
        """
        D: matrix of squared MMD
        D_sigma2: estimation of the variance of the squared MMD

        """

        N = D.shape[0]

        # Initialize with deterministic MDS
        rect_D = np.sqrt(np.maximum(D, np.zeros_like(D)))
        init = MDS(n_components=self.d, dissimilarity='precomputed').fit_transform(rect_D)
        init = init.reshape(1, N, self.d) # reshape to match hypotheses of cdist

        # Set up parameters and optimizer
        X = Parameter(torch.from_numpy(init).to(device))
        opt = torch.optim.Adam(lr=lr, params=[X])

        # Set up inputs as tensors
        D_ = torch.from_numpy(D).float().to(device)
        D_ = torch.tril(D_)
        D_ = D_.reshape(1, N, N)

        D_sigma2_ = torch.from_numpy(D_sigma2).float().to(device)
        D_sigma2_[D_sigma2_ == 0] = 0.5 * torch.min(D_sigma2_[D_sigma2_ > 0])
        D_sigma2_ = D_sigma2_.reshape(1, N, N)

        # Training loop
        loss_history = []
        with trange(n_steps) as steps_bar:
            for i in steps_bar:
                opt.zero_grad()
                l = self.loss(X, D_, D_sigma2_)
                l.backward()
                opt.step()
                steps_bar.set_postfix(loss=l.item())
                loss_history.append(l.item())

        self.loss_history = loss_history
        self.loss = loss_history[-1]
        self.X = X.detach().cpu().numpy().squeeze()

        return self.X

    def fit(self, *args, **kwargs):
        self.fit_transform(*args, **kwargs)

    def loss(self, X, D, D_sigma2):
        d_reduced = torch.cdist(X, X) ** 2
        pairwise_loglikelihood = -0.5*(d_reduced-D)**2/D_sigma2
        loglikelihood = torch.mean(torch.tril(pairwise_loglikelihood))   # Note the usage of tril 

        return -loglikelihood