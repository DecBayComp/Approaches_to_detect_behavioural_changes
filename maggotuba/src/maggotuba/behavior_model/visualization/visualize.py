import torch
import numpy as np
import numpy.random
import matplotlib as mpl
# mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mplcolors
from matplotlib.widgets import CheckButtons
from sklearn.neighbors import KernelDensity

import sklearn
import sklearn.ensemble
import sklearn.cluster
import sklearn.metrics
import sklearn.model_selection
import numpy as np
import scipy.spatial
import pandas as pd

import matplotlib.animation as animation
import multiprocessing
import itertools
import time
import logging

from scipy.spatial import Voronoi, distance_matrix
import matplotlib.cm as cm

import os
import numba

class Visualizer():
    """ 
    An engine for all graphical outputs of the training/prediction scripts

    Contains functions to deport the computations and I/O to subprocess, auxiliary functions, dimension reduction for visualization,
    functions to create animations and interactive plots, functions to create static plots.

    Since JB always has a new idea of something to plot, this class grows pretty quickly.

    Attributes
    ----------

    colors
        Class attribute. List of colors to use in the plots.
        The six first colors correspond to the color code of the behavior listed in the class attribute labels.
    labels
        Class attribute. List of the names of the behaviors. Order corresponds to the order of the colors.
    path_saved_embedder
        Optional, path to a precomputed parametric UMAP
    log_dir
        Directory in which the figures are saved
    n_points
        number of points along the spine of the larva
    target
        targets of the self-supervised objective, i.e. past, present and/or future
    dim_reduc
        algorithm to use for dimensionality reduction
    dim_embed
        dimension of the latent space
    train_losses
        list of the train losses
    eval_losses
        list of the eval losses
    train_penalties
        list of the regularization terms
    cmap
        colormap used in density plots
    processes
        internal attribute storing the processes on which the computation is deported

    Methods
    -------

    join
        Waits for computations to end
    plot_larva
        Class method. Plots a larva on a specificied axis. Returns the corresponding artists.
    update_larva
        Class method. Updates artists corresponding to a larva, using data from the argument larva
    plot_trajectory
        Launches a separate process that will execute _plot_trajectory
    _plot_trajectory
        Plots a 4x2 grid of animations showing samples and their reconstructions
    plot_moving_embed
        Launches separate processes that will execute _plot_moving_embed and _plot_moving_embed_labeled
    _plot_moving_embed
        Plots a figure with two animations. On the right are a set of larvae in the data space, and on the left the corresponding trajectory in the latent space.
        The correspondence is indicated by colors.
    _plot_moving_embed_labeled
        Plots a 2x4 grid of animations. The top row shows long trajectories in the data space and the bottom row the trajectories in the latent space.
        Trajectories are color-coded according to the behavior.
    plot_embed
        Plots the following figures :
            * unlabelled embeding : 2D plot of the latent space without any color coding
            * labeled embedding : 3x3 panels representing the 2D latent space, colored according to specific features, or according to behavior
                                  OR 3 panels representing the 3D latent space projected on xy, xz, yz axes
            * segmentation plot : 3 panels representing the labelled embedding, with every sample, or pooled in hexagons, or pooled according to a KDE
            * hexbin density : 6 panels representing a hexagonal 2D histogram of the behavios in thhe latent space
            * mixing plot : 1 plot representing the degree of mixing of the behaviors in every hexagon
            * optionally, 3d plots :
                - projection of the 3d latent space along axes xy, xz, yz
                - an interactive 3d plot where volumes of each behavior are displayed using voxels
    _prepare_unlabeled_embed
        Auxiliary function. Plots a 2D plot of the latent space without any color coding on a given axis
    _save_labeled_embed
        Auxiliary function. Plots a figure with 3x3 panels representing the 2D latent space, colored according to specific features, or according to behavior
        OR 3 panels representing the 3D latent space projected on xy, xz, yz axes
    _prepare_hexbin_plot
        Auxiliary function. Plots an hexagonal pooling of the behavior in the 2D latent space on a given axis.
    _prepare_kernel_density_segmentation_plot
        Auxiliary function. Plots the color of a most likely behavior at every point of the latent space using a KDE.
    _prepare_mixing_plot
        Auxiliary function. Plots the degree of mixing of the behaviors on an hexagonal discretization of the latent space
    _save_binned_embed_plot
        Auxiliary function calling _prepare_hexbin_plot, _prepare_kernel_density_segmentation_plot, _prepare_mixing_plot to construct and save a plot.
    _save_hexbin_behavior_density_plot
        Plots and saves a figure with 6 panels representing a hexagonal 2D histogram of the behavios in thhe latent space
    _get_convex_hull
        Wrapper for inlier detection and convex hull computation on points in a 2D space.
        Used to compute the general outline of the support of the data distribution.
    _get_contour_line
        Wrapper for level set computation. Used to compute the general outline of the support of each behavior.
    _plot_projected_3d_embed
        Plots the projection of the 3d latent space along axes xy, xz, yz
    _plot_interactive_3d_embed
        Creates an interactive plot of the 3D latent space where volumes of each behavior are displayed using voxels
        and behaviors can be toggled on and off.
    plot_losses
        Plots and saves the test and train loss as well as regularizers for the current training.
    _embed
        Launcher for dimensionality reduction methods implemented in more specialized functions
    _embed_with_pca
        Performs a PCA
    _embed_with_umap
        Performs a parametric UMAP
    _embed_with_tsne
        Performs a TSNE
    get_cmap_from_color
        Helper function to create a gradient fromwhite to the specified color. Used to plot hexbin behavior density.
    _compute_mean_label_entropy
        Coarse-grains the full dim latent space and compute the mean entropy of the labels in each cell
        Used to quantify the level of mixing.
    _compute_plot_pairwise_umap
        Plots a grid of UMAPs each computed on a subset of the data consisting of two behaviors.
        Highlights separation between behaviors.
    _compute_plot_single_behavior_umap
        Plots a grid of UMAPs each computed on a subset of the data consisting of one behavior.
        Highlights internal structure of each behavior.
    _reconstruction_error
        Plots a 3x6 grid of single behavior UMAPs colored according to the reconstruction in the past, present and future.
    _plot_vor
        Plots a single voronoi diagram on a specified axis, excluding cells of high diameter regarded as noise.
        Colors are mapped from an array of values, smoothed through a repeated diffusion operation on the voronoi diagram.
    """

    colors = ['#17202A', '#C0392B', '#8BC34A', '#2E86C1', '#26C6DA', '#F1C40F', 'orange', 'magenta']
    labels = ['run', 'bend', 'stop', 'hunch', 'back', 'roll']

    def __init__(self, log_dir, target, dim_reduc, dim_embed=2, path_saved_embedder=None):
        if path_saved_embedder is not None:
            self.path_saved_embedder = path_saved_embedder
        self.log_dir = log_dir
        self.n_points = 5
        self.target = target
        self.dim_reduc = dim_reduc
        self.dim_embed = dim_embed
        self.train_losses, self.eval_losses, self.train_penalties = [], [], []
        self.cmap = "RdYlGn"
        self.init_pool()
        self.async_results = {}
        logging.basicConfig(level=logging.INFO)
    
    def __getstate__(self):
        return {k:v for k,v in self.__dict__.items() if not(k in ['processes', 'async_results'])}
 
    def init_pool(self, nworkers=2):
        self.processes = multiprocessing.get_context('spawn').Pool(nworkers)

    def join(self):
        """Wait for all processes in self.processes to end and reset self.processes."""
        self.processes.close()
        self.processes.join()
        for n, ar in self.async_results.items():
            print(n, "Successful" if ar.successful() else "Unsuccessful")
            if not ar.successful():
                try:
                    ar.get()
                except Exception as e:
                    print(e)
        self.async_results = {}
        self.init_pool()
   
    def plot_larva(larva, behavior, ax, color=None, alpha=1.0):
        """Class method. Plots a larva on a specificied axis. Returns the corresponding artists.

        Parameters
        ----------
        larva : 2D array-like with first dimension 2
            The x-y coordinates of the points along the larva's spine, from tail to head
        behavior : int
            Index of the desired color in Visualizer.colors. If color is specified, this argument
            is still required but ignored
        ax : matplotlib Axes
            Axes on which the larva should be plotted
        color : matplotlib color specifier, optional
            Overrides the behavior parameter. Specifies a color for the larva.
            Defaults to None
        alpha : float, optional
            Alpha channel value. Defaults to 1.0

        Raises
        ------
        AssertionError
            If larva does not have the proper shape

        Returns
        -------
        larva_line : Line2D
            Connects the points along the spine of the larva 
        larva_tail : PathCollection
            Scatter plot containing one black round marker marking the tail of the larva
        larva_head : PathCollection
            Scatter plot containing one black X marker marking the head of the larva.
        larva_body : PathCollection
            Scatter plot containing round markers of the specified color at the positions of the spine.
        """
        assert(len(larva.shape)==2 and larva.shape[0]==2)
        if color is None:
            color = Visualizer.colors[behavior]
        x = larva[0]
        y = larva[1]
        larva_line, = ax.plot(x, y, linewidth=8, color=color, alpha=alpha)
        larva_tail = ax.scatter(x[:1], y[:1], color='black', s=200, alpha=alpha)
        larva_head = ax.scatter(x[-1:], y[-1:], color='black', s=250, marker='X', alpha=alpha)
        larva_body = ax.scatter(x[1:-1], y[1:-1], color=color, s=100, alpha=alpha)
        return larva_line, larva_tail, larva_head, larva_body

    def update_larva(larva, behavior, larva_line, larva_tail, larva_head, larva_body, color=None):
        """Class method. Updates artists corresponding to a larva, using data from the argument larva


        Parameters
        ----------
        larva : 2D array-like with first dimension 2
            The x-y coordinates of the points along the larva's spine, from tail to head,
            to which the artists must be updated
        behavior : int
            Index of the desired color in Visualizer.colors. If color is specified, this argument
            is still required but ignored
        larva_line : Line2D
            Connects the points along the spine of the larva 
        larva_tail : PathCollection
            Scatter plot containing one black round marker marking the tail of the larva
        larva_head : PathCollection
            Scatter plot containing one black X marker marking the head of the larva.
        larva_body : PathCollection
            Scatter plot containing round markers of the specified color at the positions of the spin
        color : matplotlib color specifier, optional
            Overrides the behavior parameter. Specifies a color for the larva.
            Defaults to None

        Raises
        ------
        AssertionError
            If larva does not have the proper shape

        Returns
        -------
        None
        """
        assert(len(larva.shape)==2 and larva.shape[0]==2)
        if color is None:
            color = Visualizer.colors[behavior]
        x = larva[0]
        y = larva[1]
        larva_line.set_data(x, y)
        larva_line.set_color(color)
        larva_tail.set_offsets(np.array([x[0], y[0]]).transpose()) 
        larva_head.set_offsets(np.array([x[-1], y[-1]]).transpose())
        larva_body.set_offsets(np.array(np.stack([x[1:-1], y[1:-1]])).transpose())
        larva_body.set_color(color)

##############################################################

    def plot_trajectory(self, data, kw='balanced_eval'):
        """Launches a separate process that will execute _plot_trajectory

        Appends the created process to self.processes.
        Multiprocessing wrapper to deport computationnal load to another process.

        Parameters
        ----------
        data : dict of array_likes with keys 'past', 'present', 'future', 'label', 'pred'
            data['past' | 'present' | 'future'] : n_larvae * x-y * time * len_larva : coordinates of the observed trajectory
            to be plotted, from x to y, start to finish, tail to head
            data['label'] : array-like of length n_larvae, supervised classifier-based behavior
            data['pred'] : n_larvae * x-y * time * len_larva : reconstruction of the coordinates
            n_larvae must be equal to 8
        kw : str, optional
            subfolder of logdir/visu in which the animation is stored
            defaults to 'balanced_eval'

        Returns
        -------
        None
        """

        logging.info("Starting thread for 'plot trajectory'.")
        self.async_results['plot_trajectory'] = self.processes.apply_async(self._plot_trajectory, (data, kw))
        
    def _plot_trajectory(self, data, kw='balanced_eval'):
        """Plots a 4x2 grid of animations showing samples and their reconstructions

        Plot short trajectories in the data space along with their reconstruction.

        Parameters
        ----------
        data : dict of array_likes with keys 'past', 'present', 'future', 'label', 'pred'
            data['past' | 'present' | 'future'] : n_larvae * x-y * time * len_larva : coordinates of the observed trajectory
            to be plotted, from x to y, start to finish, tail to head
            data['label'] : array-like of length n_larvae, supervised classifier-based behavior
            data['pred'] : n_larvae * x-y * time * len_larva : reconstruction of the coordinates
            n_larvae must be equal to 8
        kw : str, optional
            subfolder of logdir/visu in which the animation is stored
            defaults to 'balanced_eval'

        Returns
        -------
        None
        """
        truth = np.concatenate((data['past'], data['present'], data['future']),-1)

        # setup the figure
        fig, axes = plt.subplots(2, 3)
        fig.set_figheight(8)
        fig.set_figwidth(12)

        # prepare the legend
        truth_patches = [mpatches.Patch(color=Visualizer.colors[i], label=time) for i,time in enumerate(['Past','Present','Future'])]
        pred_patch = mpatches.Patch(color=Visualizer.colors[3], label='Prediction')
        fig.legend(handles=truth_patches +[pred_patch])

        # scale axes
        for i in range(6):
            ax = axes[i//3][i%3]
            ax.set_title(Visualizer.labels[int(data['label'][i])])
            ax.set_xlim(np.min(truth[i,0,2,:])-1,np.max(truth[i,0,2,:])+1)
            ax.set_ylim(np.min(truth[i,1,2,:])-1,np.max(truth[i,1,2,:])+1)

        # initialize artists
        data_artists = []
        pred_artists = []
        for i in range(6):
            ax = axes[i//3][i%3]
            data_artists.append(Visualizer.plot_larva(data['past'][i,:,:,0], 0, ax))
            if 'past' in self.target:
                pred_artists.append(Visualizer.plot_larva(data['pred'][i,:,:,0], 3, ax))
            else:
                pred_artists.append(Visualizer.plot_larva(np.zeros((2,0)), 3, ax))
        
        past_length    = data['past'].shape[-1]
        present_length = data['present'].shape[-1]
        future_length  = data['future'].shape[-1]

        def update_anim(frame):
            if frame < past_length:
                aux_frame = frame
                for i in range(6):
                    Visualizer.update_larva(data['past'][i,:,:,aux_frame], 0, *data_artists[i])
                    if 'past' in self.target:
                        Visualizer.update_larva(data['pred'][i,:,:,frame], 3, *pred_artists[i])
                    else:
                        Visualizer.plot_larva(np.zeros((2,0)), 3, *pred_artists[i])

            elif frame < past_length+present_length:
                aux_frame = frame-past_length
                for i in range(6):
                    Visualizer.update_larva(data['present'][i,:,:,aux_frame], 1, *data_artists[i])
                    if 'present' in self.target:
                        Visualizer.update_larva(data['pred'][i,:,:,frame], 3, *pred_artists[i])
                    else:
                        Visualizer.plot_larva(np.zeros((2,0)), 3, *pred_artists[i])
            else:
                aux_frame = frame-past_length-present_length
                for i in range(6):
                    Visualizer.update_larva(data['future'][i,:,:,aux_frame], 2, *data_artists[i])
                    if 'future' in self.target:
                        Visualizer.update_larva(data['pred'][i,:,:,frame], 3, *pred_artists[i])
                    else:
                        Visualizer.plot_larva(np.zeros((2,0)), 3, *pred_artists[i])
            return itertools.chain(data_artists+pred_artists)

        # save the animation
        anim = animation.FuncAnimation(fig, update_anim, frames=past_length+present_length+future_length, repeat=True, interval=200)
        anim.save(self.log_dir+'/visu/'+kw+'/larva_iter_'+str(self._iter)+'.gif')
        plt.close()

##############################################################

    def plot_embed(self, data, label, centers=None, cut=2, fit_embedder=True, kw='eval', _3d=False):
        """Plots a variety of still figures.

        Plots the following figures :
            * unlabelled embeding : 2D plot of the latent space without any color coding
            * labeled embedding : 3x3 panels representing the 2D latent space, colored according to specific features, or according to behavior
                                  OR 3 panels representing the 3D latent space projected on xy, xz, yz axes
            * segmentation plot : 3 panels representing the labelled embedding, with every sample, or pooled in hexagons, or pooled according to a KDE
            * hexbin density : 6 panels representing a hexagonal 2D histogram of the behavios in thhe latent space
            * mixing plot : 1 plot representing the degree of mixing of the behaviors in every hexagon
            * optionally, 3d plots :
                - projection of the 3d latent space along axes xy, xz, yz
                - an interactive 3d plot where volumes of each behavior are displayed using voxels

        Parameters
        ----------
        data : array-like
            array-like with 2 dimensions : n_samples * n_features
            features 0 to dim_latent-1 corresponds to coordinates in the latent space
            -5 : sum of absolute lengths of larva
            -4 : sum of absolute curvature (angle between first and third joint of the larva)
            -3 : sum of absolute length variation
            -2 : sum of absolute curvature variation
            -1 : sum of absolute central point variation (speed of the central point)
            -5 through -1 are the output of features.build_features.features_larva.
            They are centered and rescaled.
        label : array-like
            array_like with one dimension : n_samples
            Record of supervised classifier-based behavior.
        centers : array-like, optional
            array-like with 2 dimensions : n_centers * x-y
            Coordinates of centroids to be plotted in the latent space
            Usually, the centers from the K-Means regularizer.
            Defaults to None
        cut : float, optional
            limits of the plotted data along x and y
            Defaults to 10.
        fit_embedder : bool, optional
            When using a parametric embedding, such as 'UMAP', whether or not to retrain the embedder
            if an embedder is already available.
            Defaults to True
        kw : str, optional
            subfolder of logdir/visu in which the animation is stored
            defaults to 'eval'
        _3d : bool, optional
            Whether to create the projected 3d plot and the interactive 3d plot.
            The interactive plot is blocking (pauses the execution of the program until it is closed) 
            and is not saved.
            Defaults to False

        Returns
        -------
        None
        """
        #Remove feature outliers:
        idx = np.arange(data.shape[0], dtype=int)#(torch.abs(data[:,:-5])<cut).all(-1) # TODO -5 is here because the 5 last dims represent the features computed by features.py and plotted in labelled_plot_embedding
        data = data[idx]
        label = label[idx].astype(int)
        embed = data[:,:-5]
        if embed.shape[-1]>self.dim_embed:
            embed = self._embed(embed, fit=fit_embedder)
        # embed = embed.numpy()
        # data = data.numpy()

        # unlabeled embedding
        logging.info("Starting thread for 'save_unlabeled_embed'.")
        self.async_results['unlabeled_embed'] = self.processes.apply_async(self._save_unlabeled_embed, (embed, kw))

        # labeled embedding
        logging.info("Starting thread for 'save_labeled_embed'.")
        self.async_results['labeled_embed'] = self.processes.apply_async(self._save_labeled_embed, (embed, data[:,-5:], label, centers, cut, kw))

        # prepare binned plot
        logging.info("Starting thread for 'save_binned_embed'.")
        self.async_results['binned_embed'] = self.processes.apply_async(self._save_binned_embed_plot, (embed, label, kw))

        # prepare behavior density plot
        logging.info("Starting thread for 'save_hexbin_behavior_density_plot'.")
        self.async_results['hexbin_behavior_density_plot'] = self.processes.apply_async(self._save_hexbin_behavior_density_plot, (embed, label, kw))

        # mixing plot
        logging.info("Starting thread for 'mixing_plot'.")
        self.async_results['mixing_plot'] = self.processes.apply_async(self._save_mixing_plot, (embed, label, kw))

        if _3d:
            logging.info("Starting thread for 'save_projected_3d_embed'.")
            self.async_results['save_projected_3d_embed'] = self.processes.apply_async(self.plot_projected_3d_embed, (data, label, cut, kw))

    def _prepare_unlabeled_embed(self, embed, ax=None):
        if ax is None:
            ax = plt.gca()
        ax.scatter(embed[:,0], embed[:,1], alpha=0.5, s=0.5)
        ax.set_title('Unlabelled embedding')
        return ax

    def _save_unlabeled_embed(self, embed, kw='eval'):
        plt.figure()
        self._prepare_unlabeled_embed(embed)
        plt.savefig(self.log_dir+'/visu/'+kw+'/embed_iter_'+str(self._iter)+'.png')
        plt.close()

    def _save_labeled_embed(self, embed, data, label, centers=None, cut=2, kw='eval'):
        color = [Visualizer.colors[int(l)] for l in label]

        # plot with features
        fig, axes = plt.subplots(2, 3, figsize=(20,10))
        fig.suptitle(f'{self.dim_reduc} embeddings', fontsize=18)

        axes[0,0].scatter(embed[:,0], embed[:,1], c=color, alpha=1.0, s=0.3)


        # axes[0,0].legend([mpatches.Circle((0,0),1,fc=color) for color in Visualizer.colors], Visualizer.labels, fancybox=True, framealpha=0.5, fontsize=12)
        if centers is not None:
            axes[0,0].scatter(centers[:,0], centers[:,1], c='red', s=300, marker='X')
        axes[0,0].set_title('Discrete labeling', fontsize=18)
        axes[0,1].scatter(embed[:,0], embed[:,1], c=data[:,-5], cmap=self.cmap, vmin=-cut, vmax=cut, alpha=1.0, s=0.3)
        axes[0,1].set_title('Length of the larva', fontsize=18)
        axes[0,2].scatter(embed[:,0], embed[:,1], c=data[:,-4], cmap=self.cmap, vmin=-cut, vmax=cut, alpha=1.0, s=0.3)
        axes[0,2].set_title('Curvature of the larva', fontsize=18)
        axes[1,0].scatter(embed[:,0], embed[:,1], c=data[:,-3], cmap=self.cmap, vmin=-cut, vmax=cut, alpha=1.0, s=0.3)
        axes[1,0].set_title('Length variation', fontsize=18)
        axes[1,1].scatter(embed[:,0], embed[:,1], c=data[:,-2], cmap=self.cmap, vmin=-cut, vmax=cut, alpha=1.0, s=0.3)
        axes[1,1].set_title('Curvature variation', fontsize=18)
        axes[1,2].scatter(embed[:,0], embed[:,1], c=data[:,-1], cmap=self.cmap, vmin=-cut, vmax=cut, alpha=1.0, s=0.3)
        axes[1,2].set_title('Speed of the central point', fontsize=18)
        norm = mpl.colors.Normalize(vmin=-cut, vmax=cut)
        fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=self.cmap), ax=axes)
        for ax in axes.flatten():
            ax.set_xticks([])
            ax.set_yticks([])

        plt.savefig(self.log_dir+'/visu/'+kw+'/labeled_embed_w_features_iter_'+str(self._iter)+'.png', dpi=100)
        plt.close()

        # plot without features
        plt.figure(figsize=(4,4))
        plt.scatter(embed[:,0], embed[:,1], c=color, alpha=1.0, s=0.3)
        plt.tight_layout()
        plt.axis('off')
        plt.savefig(self.log_dir+'/visu/'+kw+'/labeled_embed_iter_'+str(self._iter)+'.png', dpi=300)


    def _prepare_hexbin_plot(self, embed, label, ax=None):
        if ax is None:
            ax = plt.gca()
        cmap = mplcolors.ListedColormap([Visualizer.colors[i] for i in range(6)])
        def reduce_func(C):
            val, counts_ = np.unique(C, return_counts=True)
            counts = np.zeros(6)
            for v, c in zip(val, counts_):
                counts[int(v)] = c
            counts /= np.array([0.25, 0.25, 0.125, 0.125, 0.125, 0.125])
            return np.argmax(counts)/6+.5

        ax.hexbin(embed[:,0], embed[:,1], C=label, cmap=cmap, gridsize=15, reduce_C_function=reduce_func)
        ax.scatter(embed[:,0], embed[:,1], c=label, edgecolors='grey', s=8, cmap=cmap)
        
    def _prepare_kernel_density_segmentation_plot(self, embed, label, ax=None):
        if ax is None:
            ax = plt.gca()
        # prepare Voronoi-like plot
        cmap = mplcolors.ListedColormap([Visualizer.colors[i] for i in range(6)])
        kde = [KernelDensity() for i in range(6)]

        for l, estimator in enumerate(kde):
            estimator.fit(embed[label==l])
        XX, YY = np.meshgrid(np.linspace(-3,3,100), np.linspace(-3,3,100))

        def score(x,y):
            return np.argmax(np.stack([estimator.score_samples(np.array([[x, y]])).item() for estimator in kde]))
                        
        vect_score = np.vectorize(score)
        xmin, xmax = np.min(embed[:,0])-1., np.max(embed[:,0])+1.0
        ymin, ymax = np.min(embed[:,1])-1., np.max(embed[:,1])+1.0
        XX, YY = np.meshgrid(np.linspace(xmin,xmax,250), np.linspace(ymin,ymax,250))
        scores = vect_score(XX.reshape(-1), YY.reshape(-1)).reshape(250,250)

        ax.imshow(scores, cmap=cmap, extent=(xmin,xmax,ymin,ymax), origin='lower')
        ax.scatter(embed[:,0], embed[:,1], c=label, edgecolors='grey', s=8, cmap=cmap)

    def _prepare_mixing_plot(self, embed, label, ax=None, colorbar=True):
        if ax is None:
            ax = plt.gca()
        cmap = 'RdYlGn_r'
        def reduce_func(C):
            val, counts_ = np.unique(C, return_counts=True)
            p = counts_/np.sum(counts_)
            entropy = -np.mean(p*np.log(p))
            return entropy

        hb = ax.hexbin(embed[:,0], embed[:,1], C=label, cmap=cmap, gridsize=15, reduce_C_function=reduce_func)
        plt.colorbar(mappable=hb, ax=ax)

    def _save_mixing_plot(self, embed, label, kw='eval'):
        plt.figure()
        self._prepare_mixing_plot(embed, label)
        plt.title('Local label entropy')
        plt.savefig(self.log_dir+'/visu/'+kw+'/mixing_plot_iter_'+str(self._iter)+'.png')
        plt.close()

    def _save_binned_embed_plot(self, embed, label, kw='eval'):
        _, ax = plt.subplots(1,2,figsize=(20,10))

        color = [Visualizer.colors[int(l)] for l in label]
        cmap = mplcolors.ListedColormap([Visualizer.colors[i] for i in range(6)])
        ax[0].set_title('Discrete labeling', fontsize=30)
        ax[0].scatter(embed[:,0], embed[:,1], c=color, cmap=cmap, alpha=1.0, s=15)

        # prepare hexbin plot
        self._prepare_hexbin_plot(embed, label, ax[1])

        # prepare Voronoi-like plot
        # self._prepare_kernel_density_segmentation_plot(embed, label, ax[2])

        plt.savefig(self.log_dir+'/visu/'+kw+'/binned_embed_iter_'+str(self._iter)+'.png')
        plt.close()

    def _save_hexbin_behavior_density_plot(self, embed, label, kw='eval'):
        fig, axs = plt.subplots(2,3, figsize=(15,10), sharex=True, sharey=True)

        hull = self._get_convex_hull(embed)
        np.savez(self.log_dir+'/visu/'+kw+'/convex_hull_iter_{}.npz'.format(self._iter), hull=hull)
        for l, ax in enumerate(axs.flatten()):
            coords = embed[label==l,:]
            cmap = Visualizer.get_cmap_from_color(l)
            hb = ax.hexbin(coords[:,0], coords[:,1], gridsize=15, cmap=cmap)
            pbar = plt.colorbar(mappable=hb, ax=ax)
            pbar.set_ticks([])
            title = '''Density of "{}" in the latent space'''.format(Visualizer.labels[l])
            ax.set_title(title)
            ax.plot(hull[:,0], hull[:,1], ls='-', c='grey', lw=2)

            # contours = self._get_contour_line(ax, embed, label, l)
            # np.savez(self.log_dir+'/visu/'+kw+'/contours_{}.npz'.format(Visualizer.labels[l]), *contours.allsegs[0])

        plt.savefig(self.log_dir+'/visu/'+kw+'/hexbin_behavior_density_'+str(self._iter)+'.png')
        plt.close()

    def _get_convex_hull(self, points):
        from sklearn.ensemble import IsolationForest
        from scipy.spatial import ConvexHull
        labels = IsolationForest().fit_predict(points)
        inliers = points[labels==1,:]
        hull = ConvexHull(inliers)
        return inliers[np.concatenate([hull.vertices, hull.vertices[:1]]),:]

    def _get_contour_line(self, ax, points, labels, behavior=None):
        estimator = KernelDensity()
        if behavior is not None:
            points = points[labels==behavior]
            col = Visualizer.colors[behavior]
        else:
            col = 'grey'
        estimator.fit(points)
        xmin, xmax = np.min(points[:,0]), np.max(points[:,0])
        ymin, ymax = np.min(points[:,1]), np.max(points[:,1])

        xx, yy = np.meshgrid(np.linspace(xmin, xmax, 100), np.linspace(ymin, ymax, 100))
        xy = np.hstack([xx.reshape(-1,1), yy.reshape(-1,1)])
        z = np.exp(estimator.score_samples(xy).reshape((100,100)))        

        contours = ax.contour(xx, yy, z, levels=[0.008], colors=col)
        ax.clabel(contours, inline=1)

        return contours
        

            
   

##############################################################

    def plot_projected_3d_embed(self, data, label, cut=10, kw='eval'):
        #Remove feature outliers:
        idx = (np.abs(data[:,:-5])<cut).all(-1) # TODO -5 is here because the 5 last dims represent the features computed by features.py and plotted in labelled_plot_embedding
        data = data[idx]
        embed = data[:,:-5]
        # label = label[idx].numpy().astype(int)
        if embed.shape[-1]>3:
            embed = self._embed_with_umap(embed, fit=True, dim_embed=3)
        embed = embed

        color = [Visualizer.colors[int(l)] for l in label]
        cmap = mplcolors.ListedColormap([Visualizer.colors[i] for i in range(6)])

        fig, axs = plt.subplots(1,3,figsize=(30,10))
        for ax, idx, axesname in zip(axs, [[0,1],[0,2], [1,2]], ['XY', 'XZ', 'YZ']):
            ax.scatter(embed[:,idx[0]], embed[:,idx[1]], c=color, alpha=1.0, s=15)
            ax.set_title(axesname)
        fig.suptitle('3D UMAP')
        plt.savefig(self.log_dir+'/visu/'+kw+'/3d_embed_iter_'+str(self._iter)+'.png')
        plt.close()

    def plot_interactive_3d_embed(self, data, label, cut=10, kw='eval'):
        #Remove feature outliers:
        idx = (np.abs(data[:,:-5])<cut).all(-1) # TODO -5 is here because the 5 last dims represent the features computed by features.py and plotted in labelled_plot_embedding
        data = data[idx]
        embed = data[:,:-5]
        if embed.shape[-1]>3:
            embed = self._embed_with_umap(embed, fit=False, dim_embed=3)
        embed = embed

        # compute inliers
        inliers = np.zeros_like(label).astype(bool)
        from sklearn.ensemble import IsolationForest
        for l in range(6):
            inliers_l = (IsolationForest().fit_predict(embed[label==l])==1)
            inliers[label==l] = inliers_l
        
        button_labels = Visualizer.labels
        colors = Visualizer.colors

        # compute histograms
        # gridsize
        N = 40
        # coordinates
        span = np.linspace(-cut,cut,N)
        xx, yy, zz = np.meshgrid(span, span, span)

        histogram = np.zeros((6, N-1,N-1,N-1), dtype=int)
        for p, l, i in zip(embed, label, inliers):
            if i:
                i = np.searchsorted(span[:-1], p[0])-1
                j = np.searchsorted(span[:-1], p[1])-1
                k = np.searchsorted(span[:-1], p[2])-1
                histogram[l,i,j,k] += 1


        # create figure
        fig = plt.figure()

        # Create 3d plot
        ax3d = plt.axes((0.02, 0.02, 0.76, 0.96), projection='3d')
        plots = []
        for l, c in zip(range(6), colors):
            voxels = ax3d.voxels(xx, yy, zz, histogram[l]>1, facecolors=c, edgecolors=None)
            plots.append(voxels)

        # Create checkbox
        axCheckbox = plt.axes((0.82, 0.3, 0.16, 0.4))
        checkbox = CheckButtons(axCheckbox, button_labels, actives=[True for l in button_labels])

        # create checkbox behavior
        def on_click(label):
            index = button_labels.index(label)
            for face in plots[index].values():
                face.set_visible(not face.get_visible())
            plt.draw()

        # connect behavior to checkbox
        checkbox.on_clicked(on_click)

        # plt.show()
        
##############################################################

    def plot_losses(self, kw='eval'):
        l = len(self.train_losses)
        x_train = np.linspace(0, l, l)
        x_eval = np.linspace(0, l, len(self.eval_losses))
        plt.plot(x_train, self.train_losses, label='train loss')
        plt.plot(x_eval, self.eval_losses, label='eval loss')
        evalloss = min(self.eval_losses)
        plt.title(f'Iter {self._iter} ; Best Eval Loss : {str(evalloss)[:6]}')
        plt.axhline(evalloss, linestyle='--')
        plt.legend()
        plt.savefig(self.log_dir+'/visu/'+kw+'/losses'+'.png')
        plt.close()

##############################################################

    def _embed(self, data, fit=True, filename='parametric_umap'):
        '''Reduces the dimensionality of the latent space using a predefined method'''
        if self.dim_reduc == 'PCA':
            logging.info('Reducing dimension using pca')
            return self._embed_with_pca(data, fit)
        elif self.dim_reduc == 'UMAP':
            logging.info('Reducing dimension using umap')
            return self._embed_with_umap(data, fit, filename=filename)
        elif self.dim_reduc == 'TSNE':
            logging.info('Reducing dimension using TSNE')
            return self._embed_with_tsne(data, fit)
        else:
            raise Exception('Projection method not implemented')

    def _compute_pairwise_embed(self, embeddings, labels, fit=True, balance=True):
        # saves pairwise umaps
        # parallelizes the embedding
        dimreduc_dict = {}
        counter = 0
        for label1 in range(5):
            for label2 in range(label1+1, 6):
                counter += 1
                logging.info('Computing pairwise UMAP #{}/15, {} against {}.'.format(counter, Visualizer.labels[label1], Visualizer.labels[label2]))
                if not(balance):
                    embeds = embeddings[np.logical_or(labels==label1, labels==label2)]
                    labels2d = labels[np.logical_or(labels==label1, labels==label2)]
                else:
                    embeds1 = embeddings[labels==label1]
                    embeds2 = embeddings[labels==label2]
                    n_embeds = min(len(embeds1), len(embeds2))
                    embeds1 = embeds1[numpy.random.permutation(len(embeds1))[:n_embeds]]
                    embeds2 = embeds2[numpy.random.permutation(len(embeds2))[:n_embeds]]
                    embeds = np.vstack([embeds1, embeds2])
                    labels2d = np.hstack([label1*np.ones(n_embeds, dtype=int), label2*np.ones(n_embeds, dtype=int)])
                embeds2d = self._embed(embeds,fit=fit,filename='{}_{}_parametric_umap'.format(Visualizer.labels[label1], Visualizer.labels[label2]))
                dimreduc_dict[(label1, label2)] = (embeds2d, labels2d)

        return dimreduc_dict

    def _compute_single_embed(self, embeddings, labels, fit=True):
        # saves single behaviour umaps
        # parallelizes the embedding
        dimreduc_dict = {}
        for label in range(6):
            logging.info('Computing single behavior UMAP #{}/6, {}.'.format(label+1, Visualizer.labels[label]))
            embeds = embeddings[labels==label]
            labels2d = labels[labels==label]
            embeds2d = self._embed(embeds,fit=fit, filename='{}_parametric_umap'.format(Visualizer.labels[label]))
            dimreduc_dict[label] = (embeds2d, labels2d)
       
        return dimreduc_dict

    def _embed_with_pca(self, data, fit=True, **kwargs):
        from sklearn.decomposition import PCA
        pca = PCA(n_components=self.dim_embed)
        return pca.fit_transform(data)

    def _embed_with_umap(self, data, fit=True, dim_embed=None, filename='parametric_umap'):
        from umap.parametric_umap import ParametricUMAP, load_ParametricUMAP
        import tensorflow as tf

        tf_data = tf.convert_to_tensor(data)
        if fit:
            os.makedirs(os.path.join(self.log_dir, 'umap'), exist_ok=True)
            numba.set_num_threads(8)
            logging.info('fitting the embedder')
            if dim_embed is None:
                dim_embed = self.dim_embed
            UMAP = ParametricUMAP(n_components=dim_embed)
            UMAP.fit(tf_data)
            UMAP.save(self.log_dir+'/umap/'+filename)
            fig, ax = plt.subplots()
            ax.plot(UMAP._history['loss'])
            ax.set_ylabel('Cross Entropy')
            ax.set_xlabel('Epoch')
            fig.suptitle('Parametric UMAP training loss', fontsize=20)
            plt.savefig(self.log_dir+'/umap/'+filename+'_loss_'+str(self._iter)+'.png')
            plt.close()
        else:
            logging.info('not fitting the embedder')
            if hasattr(self, 'path_saved_embedder') and filename == 'parametric_umap':
                UMAP = load_ParametricUMAP(self.path_saved_embedder)
            else:
                UMAP = load_ParametricUMAP(self.log_dir+'/umap/'+filename)
        return UMAP.transform(tf_data)

    def _embed_with_tsne(self, data, fit=True, **kwargs):
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=self.dim_embed)
        return tsne.fit_transform(data)

    def get_cmap_from_color(color_idx, length=256):
        color_str = Visualizer.colors[color_idx]
        if color_str == 'black':
            stacked_hsv = np.zeros((length,3))
            stacked_hsv[::-1,2] = np.linspace(0,1,length)
        else:
            color = mplcolors.rgb_to_hsv(mplcolors.to_rgb(color_str)).reshape(1,3)
            color[0,2] = 1.0
            stacked_hsv = np.tile(color, (length, 1))
            stacked_hsv[:,1] = np.linspace(0,1,length)

        stacked_rgb = mplcolors.hsv_to_rgb(stacked_hsv)
        cmap = mplcolors.ListedColormap(stacked_rgb, name=color_str)

        return cmap

    def _compute_mean_label_entropy(self, embeddings, labels, n_cells=50, kw='eval'):
        kmeansEngine = sklearn.cluster.KMeans(n_clusters=n_cells)
        coarseGrainedEmbeddings = kmeansEngine.fit_predict(embeddings)
        entropies = []
        for cell in range(n_cells):
            labels_in_cell = labels[coarseGrainedEmbeddings==cell]
            _, counts = np.unique(labels_in_cell, return_counts=True)
            distrib = counts/np.sum(counts)
            entropy = -np.sum(distrib*np.log(distrib))
            entropies.append(entropy)

        _, ax = plt.subplots(1,2,figsize=(20,10))

        ax[0].hist(entropies, bins=10)
        ax[0].axvline(np.mean(entropies), c='r', ls='--')
        ax[0].set_title("Entropy of labels over a {}-cells coarse graining of the latent space. Average : {:.2e}".format(n_cells, np.mean(entropies)))

        ax[1].hist([len(labels[coarseGrainedEmbeddings==cell]) for cell in range(n_cells)])
        ax[1].set_title('Number of data point per cell')

        plt.savefig(self.log_dir+'/visu/'+kw+'/coarse_label_entropy_'+str(self._iter)+'.png', dpi=100)
        plt.close()

    # Various UMAPs on subsets of the data.
    def _compute_plot_pairwise_umap(self, embeddings, labels, balance=True, kw='eval', fit=True):
        dimreduc_dict = self._compute_pairwise_embed(embeddings, labels, balance=balance, fit=fit)

        # Plot the matrix
        fig, axs = plt.subplots(5,5, figsize=(18,18))
        for label1 in range(5):
            for label2 in range(label1+1, 6):
                embeds2d, labels2d = dimreduc_dict[(label1, label2)]
                axs[label1,label2-1].scatter(embeds2d[:,0], embeds2d[:,1], c=[Visualizer.colors[l] for l in labels2d], alpha=1.0, s=0.3)
                axs[label1, label2-1].set_xticks([])
                axs[label1, label2-1].set_yticks([])

        for i in range(1, 5):
            for j in range(i):
                axs[i,j].axis('off')

        for i in range(5):
            for j in range(i+1,6):
                axs[0,j-1].set_title(Visualizer.labels[j])
                axs[j-1,i].set_ylabel(Visualizer.labels[i])
        fig.suptitle('Pairwise UMAP embedding')
        plt.tight_layout()
        plt.savefig(self.log_dir+'/visu/'+kw+'/pairwise_umap_plot_0_'+'.png')
        plt.close()

        # Plot specific pairwise embeddings
        for label1, label2 in [(0,1), (0,5), (3,4)]:
            embeds2d, labels2d = dimreduc_dict[(label1, label2)]
            plt.figure(figsize=(4,4))
            plt.scatter(embeds2d[:,0], embeds2d[:,1], c=[Visualizer.colors[l] for l in labels2d], alpha=1.0, s=0.3)
            plt.tight_layout()
            plt.axis('off')
            plt.savefig(self.log_dir+'/visu/'+kw+f'/umap_{Visualizer.labels[label1]}_{Visualizer.labels[label2]}'+'.png', dpi=300)
            plt.close()

    def _compute_plot_single_behaviour_umap(self, embeddings, labels, fit=True, kw='eval'):
        dimreduc_dict = self._compute_single_embed(embeddings, labels, fit=fit)
        fig, axs = plt.subplots(2,3)
        for label, ax in zip(range(6), axs.flatten()):
            embeds2d, labels2d = dimreduc_dict[label]
            # cmap = Visualizer.get_cmap_from_color(label)
            # ax.hexbin(embeds2d[:,0], embeds2d[:,1], gridsize=15, cmap=cmap)
            ax.scatter(embeds2d[:,0], embeds2d[:,1], c=Visualizer.colors[label], alpha=1.0, s=0.3)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(Visualizer.labels[label])

        fig.suptitle('Single behaviour UMAP embedding')
        plt.savefig(self.log_dir+'/visu/'+kw+'/single_behaviour_umap_plot_0_'+'.png', dpi=100)
        plt.close()
    # Plots of the reconstruction error in the latent space
    def _reconstruction_error(self, embeds, labels, past_rmse, present_rmse, future_rmse, kw='eval'):
        max_rmse = np.max(np.hstack([past_rmse, present_rmse, future_rmse]))
        norm_past_rmse = past_rmse/max_rmse
        norm_present_rmse = present_rmse/max_rmse
        norm_future_rmse = future_rmse/max_rmse

        dimreduc_dict = self._compute_single_embed(embeds, labels, fit=False)

        # per behavior reconstruction error with per behavior colorscale
        fig, axs = plt.subplots(3,6, figsize=(20,10))
        scalar_mappable = cm.ScalarMappable(cmap='viridis')
        for label in range(6):
            embeds2d, _ = dimreduc_dict[label]
            # label_max_rmse = np.max(np.concatenate([norm_past_rmse[labels==label], norm_present_rmse[labels==label], norm_future_rmse[labels==label]]))
            for row, rmse_arr in enumerate([norm_past_rmse, norm_present_rmse, norm_future_rmse]):
                scalar_mappable = Visualizer._plot_vor(axs[row,label], embeds2d, rmse_arr[labels==label])
                scalar_mappable.set_array([0, np.max(rmse_arr[labels==label])*max_rmse])
                axs[row,label].set_xticks([])
                axs[row, label].set_yticks([])
                axs[row, label].set_facecolor('k')
                plt.colorbar(scalar_mappable, ax=axs[row, label])
        for column, label in enumerate(Visualizer.labels):
            axs[0, column].set_title(label)
        for row, label in enumerate(['Past', 'Present', 'Future']):
            axs[row,0].set_ylabel(label)

        fig.suptitle('Reconstruction error on the past, present and future, mapped to the latent space')
        plt.savefig(self.log_dir+'/visu/'+kw+'/latent_rmse'+'.png', dpi=100)
        plt.close()

    # Plot a smoothed colored voronoi graph.
    def _plot_vor(ax, points, vals):


        vor = Voronoi(points)
        ax.set_facecolor('k')
        scalar_mappable = cm.ScalarMappable(cmap='viridis')

        # find cutoff diameter
        def diameter_polygon(poly):
            stacked_poly = np.vstack(poly)
            dist_mat = distance_matrix(stacked_poly, stacked_poly)

            return np.max(dist_mat)

        diams = []
        for region in vor.regions:
            if not(-1 in region or region == []):
                polygon = [vor.vertices[i] for i in region]
                diams.append(diameter_polygon(polygon))
        threshold_diam = sorted(diams)[int(0.95*len(diams))]

        # smooth the voronoi graph
        def get_neighbors(voronoi):
            adjacency_list = [[] for i in voronoi.points]
            for p1, p2 in voronoi.ridge_points:
                adjacency_list[p1].append(p2)
                adjacency_list[p2].append(p1)
            return adjacency_list

        def smooth(Y, adjacency_list, alpha=0.35):
            smoothY = np.empty_like(Y)
            for idx, neighbors in enumerate(adjacency_list):
                smoothY[idx] = (1.-alpha)*Y[idx]+alpha*np.mean(Y[neighbors])
            return smoothY

        adjacency_list = get_neighbors(vor)
        for i in range(6):
            vals = smooth(vals, adjacency_list)
        vals = smooth(vals, adjacency_list)

        # normalize values to plotted range
        plotted_val = []
        for idx, v in enumerate(vals):
            if vor.point_region[idx] != -1:
                if not -1 in vor.regions[vor.point_region[idx]]:
                    polygon = [vor.vertices[i] for i in vor.regions[vor.point_region[idx]]]
                    if diameter_polygon(polygon) < threshold_diam:
                        plotted_val.append(v)
        max_v = max(plotted_val)
        min_v = min(plotted_val)
        vals = (vals-min_v)/(max_v-min_v)

        # plot the polygons
        for idx, v in enumerate(vals):
            if vor.point_region[idx] != -1:
                if not -1 in vor.regions[vor.point_region[idx]]:
                    # dealing with finite regions
                    polygon = [vor.vertices[i] for i in vor.regions[vor.point_region[idx]]]
                    if diameter_polygon(polygon) < threshold_diam:
                        polygon_x = [p[0] for p in polygon]
                        polygon_y = [p[1] for p in polygon]
                        ax.fill(polygon_x, polygon_y, color=scalar_mappable.to_rgba(v, norm=False))
        return scalar_mappable

    #################################

    def _transition_map(self, embeds, labels, future_labels, res=100, kw='eval', kind='all'):
        if kind == 'runbend':
            nlabels = 2
            figsize = (8,4)
        else:
            figsize = (20,10)
            nlabels = 6

        dimreduc_dict = self._compute_single_embed(embeds, labels, fit=False)

        fig, axs = plt.subplots(nlabels, nlabels, figsize=figsize)
        norm = mplcolors.Normalize(0, 1)
        for label in range(nlabels):
            sub_future_labels = future_labels[labels==label]
            embeds2d, _ = dimreduc_dict[label]
            embeds2d = embeds2d
            xmin, ymin, xmax, ymax = *np.min(embeds2d, axis=0), *np.max(embeds2d, axis=0)
            h, w = ymax-ymin, xmax-xmin
            xmin -= 0.05*w
            ymin -= 0.05*h
            xmax += 0.05*w
            ymax += 0.05*h

            xgrid = np.linspace(xmin, xmax, res)
            ygrid = np.linspace(ymin, ymax, res)
            xx, yy = np.meshgrid(xgrid, ygrid)
            xx = xx.reshape(-1,1)
            yy = yy.reshape(-1,1)
            xxyy = np.hstack([xx, yy])

            dist_matrix = scipy.spatial.distance_matrix(embeds2d, embeds2d)

            kernel_bandwidth = 0.5*(h+w)/20

            full_kde = sklearn.neighbors.KernelDensity(bandwidth=kernel_bandwidth).fit(embeds2d)

            grid_density = np.log(len(embeds2d)) + full_kde.score_samples(xxyy)

            not_in_support, edges = Visualizer.trim_grid(xxyy, embeds2d)

            for future_label in range(nlabels):
                future_label_embed = embeds2d[sub_future_labels==future_label]
                if future_label_embed.size > 0:
                    label_kde = sklearn.neighbors.KernelDensity(bandwidth=kernel_bandwidth).fit(future_label_embed)
                    grid_label_density = np.log(len(future_label_embed)) + label_kde.score_samples(xxyy)
                else:
                    grid_label_density = np.full(len(xx), fill_value=-20)

                transition_proba = np.exp(grid_label_density-grid_density)
                masked_transition_proba = np.ma.masked_where(not_in_support, transition_proba)
                masked_transition_proba = masked_transition_proba.reshape(res, res)


                axs[label, future_label].imshow(masked_transition_proba,
                                                origin='lower', extent=(xmin, xmax, ymin, ymax), aspect='auto', vmin=0, vmax=1)
                axs[label, future_label].set_xticks([])
                axs[label, future_label].set_yticks([])

                axs[label, future_label].set_facecolor('k')
        for label in range(nlabels):
            axs[label, 0].set_ylabel(Visualizer.labels[label])
            axs[0, label].set_title(Visualizer.labels[label])

        scalar_mappable = cm.ScalarMappable(cmap='viridis', norm=mplcolors.Normalize(vmin=0, vmax=1))
        plt.colorbar(scalar_mappable, ax=axs.ravel(), ticks=[0,.5,1.])
        plt.savefig(self.log_dir+'/visu/'+kw+f'/transition_map_{kind}'+'.png', dpi=100)
        plt.close()

    def _transition_proba(self, embeds, labels, future_labels, res=100, kw='eval'):
        scalar_mappable = cm.ScalarMappable(cmap='viridis')

        embeds2d = self._embed(embeds, fit=False)

        plt.figure()
        norm = mplcolors.Normalize(0, 1)

        xmin, ymin, xmax, ymax = *np.min(embeds2d, axis=0), *np.max(embeds2d, axis=0)
        h, w = ymax-ymin, xmax-xmin
        xmin -= 0.05*w
        ymin -= 0.05*h
        xmax += 0.05*w
        ymax += 0.05*h

        xgrid = np.linspace(xmin, xmax, res)
        ygrid = np.linspace(ymin, ymax, res)
        xx, yy = np.meshgrid(xgrid, ygrid)
        xx = xx.reshape(-1,1)
        yy = yy.reshape(-1,1)
        xxyy = np.hstack([xx, yy])

        kernel_bandwidth = 0.5*(h+w)/20

        full_kde = sklearn.neighbors.KernelDensity(bandwidth=kernel_bandwidth).fit(embeds2d)
        grid_density = np.log(len(embeds2d)) + full_kde.score_samples(xxyy)
        not_in_support, edges = Visualizer.trim_grid(xxyy, embeds2d)

        transition_label = (labels != future_labels)
        transition_embed = embeds2d[transition_label]
        transition_kde = sklearn.neighbors.KernelDensity(bandwidth=kernel_bandwidth).fit(transition_embed)
        grid_transition_density = np.log(np.sum(transition_label)) + transition_kde.score_samples(xxyy)



        transition_proba = np.exp(grid_transition_density-grid_density)
        masked_transition_proba = np.ma.masked_where(not_in_support, transition_proba)
        masked_transition_proba = masked_transition_proba.reshape(res, res)
        scalar_mappable.set_array([0, np.max(masked_transition_proba)])


        plt.imshow(masked_transition_proba, origin='lower', extent=(xmin, xmax, ymin, ymax), aspect='auto')
        plt.xticks([])
        plt.yticks([])
        plt.colorbar(scalar_mappable)
        plt.gca().set_facecolor('k')

        plt.savefig(self.log_dir+'/visu/'+kw+'/transition_proba'+'.png', dpi=100)
        plt.close()
    #####################################

    def train_evaluation_classifier(self, X, y, kw='eval'):
        clf = sklearn.ensemble.RandomForestClassifier(criterion='entropy', max_depth=20)
        cross_validated_labels = sklearn.model_selection.cross_val_predict(clf, X, y, cv=10)
        confusion_matrix = sklearn.metrics.confusion_matrix(y, cross_validated_labels)

        # plot the confusion matrix
        disp = sklearn.metrics.ConfusionMatrixDisplay.from_predictions(y, cross_validated_labels, display_labels=Visualizer.labels, normalize='true')
        disp.plot()
        plt.savefig(self.log_dir+'/visu/'+kw+'/confusion_matrix'+'.png', dpi=100)

        # plt.show()

        # plot the feature importance
        '''
        clf.fit(X,y)
        importances = clf.feature_importances_
        std = np.std([tree.feature_importances_ for tree in clf.estimators_], axis=0)

        feature_names = [f"feature {i}" for i in range(X.shape[1])]
        forest_importances = pd.Series(importances, index=feature_names)

        fig, ax = plt.subplots()
        forest_importances.plot.bar(yerr=std, ax=ax)
        ax.set_title("Feature importances using MDI")
        ax.set_ylabel("Mean decrease in impurity")
        fig.tight_layout()
        # plt.show()
        plt.close()'''


    ########################################

    def alpha_shape(points, alpha, only_outer=True):
        """
        Compute the alpha shape (concave hull) of a set of points.
        https://stackoverflow.com/questions/23073170/calculate-bounding-polygon-of-alpha-shape-from-the-delaunay-triangulation
        :param points: np.array of shape (n,2) points.
        :param alpha: alpha value.
        :param only_outer: boolean value to specify if we keep only the outer border
        or also inner edges.
        :return: set of (i,j) pairs representing edges of the alpha-shape. (i,j) are
        the indices in the points array.
        """
        assert points.shape[0] > 3, "Need at least four points"

        def add_edge(edges, i, j):
            """
            Add a line between the i-th and j-th points,
            if not in the list already
            """
            if (i, j) in edges or (j, i) in edges:
                # already added
                assert (j, i) in edges, "Can't go twice over same directed edge right?"
                if only_outer:
                    # if both neighboring triangles are in shape, it is not a boundary edge
                    edges.remove((j, i))
                return
            edges.add((i, j))

        tri = scipy.spatial.Delaunay(points)
        edges = set()
        # Loop over triangles:
        # ia, ib, ic = indices of corner points of the triangle
        for ia, ib, ic in tri.simplices:
            pa = points[ia]
            pb = points[ib]
            pc = points[ic]
            # Computing radius of triangle circumcircle
            # www.mathalino.com/reviewer/derivation-of-formulas/derivation-of-formula-for-radius-of-circumcircle
            a = np.sqrt((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2)
            b = np.sqrt((pb[0] - pc[0]) ** 2 + (pb[1] - pc[1]) ** 2)
            c = np.sqrt((pc[0] - pa[0]) ** 2 + (pc[1] - pa[1]) ** 2)
            s = (a + b + c) / 2.0
            area = np.sqrt(s * (s - a) * (s - b) * (s - c))
            circum_r = a * b * c / (4.0 * area)
            if circum_r < alpha:
                add_edge(edges, ia, ib)
                add_edge(edges, ib, ic)
                add_edge(edges, ic, ia)
        return edges

    def check_is_inside(point_to_test, data_points, edges):
        '''
        https://stackoverflow.com/questions/217578/how-can-i-determine-whether-a-2d-point-is-within-a-polygon
        '''
        xmin, ymin = np.min(data_points, axis=0)
        xmax, ymax = np.max(data_points, axis=0)
        xtest, ytest = point_to_test

        if (xtest < xmin or xtest > xmax or ytest < ymin or ytest > ymax):
            return False

        n_intersect_from_left = 0
        for i, j in edges:
            p1 = data_points[i]
            p2 = data_points[j]
            if ytest >= min(p1[1], p2[1]) and ytest <= max(p1[1], p2[1]):
                # there is an intersection, compute its abscissa
                left_point = p1 if p1[0] < p2[0] else p2
                right_point = p2 if p1[0] < p2[0] else p1
                xinter = left_point[0] + (ytest-left_point[1])/(right_point[1]-left_point[1])*(right_point[0]-left_point[0])

                # if the intersection is left of the point tested, count it
                if xinter <= point_to_test[0]:
                    n_intersect_from_left += 1
        
        return bool(n_intersect_from_left%2)

    def trim_grid(grid, data_points):
        edges = Visualizer.alpha_shape(data_points, alpha=1.0)
        mask = np.full(len(grid), fill_value=False)
        for i, p in enumerate(grid):
            mask[i] = not(Visualizer.check_is_inside(p, data_points, edges))
        return mask, edges

    ##################################################
    def behavior_reclustering(self, trajs, embeds, labels, to_cluster=3, fit=False, kw='eval'):
        embeds = embeds[labels==to_cluster]


        from gudhi.clustering.tomato import Tomato
        tomato = Tomato(k=10)

        tomato.fit(embeds)
        tomato.plot_diagram()
        if to_cluster == 5:
            n_clusters = 3
        elif to_cluster == 2:
            n_clusters = 3
        else:
            n_clusters = int(input('nclusters : '))
        tomato.n_clusters_ = n_clusters
        new_cluster_labels = tomato.labels_
        plt.figure()
        plt.scatter(embeds2d[:,0], embeds2d[:,1], c=new_cluster_labels)
        # plt.show()
        for cluster_idx in range(n_clusters):
            pass
            # folder_path = os.path.join(self.log_dir, 'reclustering', Visualizer.labels, target, dim_reduc))
        plt.savefig(self.log_dir+'/reclustering/'+kw+'/transition_proba'+'.png')
        plt.close()

        # if to_cluster == 2:
        #     # compute persistence diagram of h1 homology of stops
        #     from gudhi import RipsComplex, plot_persistence_barcode, plot_persistence_diagram
        #     rips_complex = RipsComplex(points=embeds)
        #     simplex_tree = rips_complex.create_simplex_tree(max_dimension=2)
        #     persistence_diagram = simplex_tree.persistence()
        #     plot_persistence_barcode(persistence_diagram)
        #     plt.show()
        #     plot_persistence_diagram(persistence_diagram)
        #     plt.show()
