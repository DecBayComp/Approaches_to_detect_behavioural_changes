from maggotuba.mmd import computeWitnessFunctionOnMesh
import os
import json
import numpy as np
import matplotlib.pyplot as plt

# copy-pasted and then modified from cli.cli_plot

def add_arguments_compare_lines(parser):
    parser.add_argument('--name', '-n', type=str, default=None)
    parser.add_argument('--line1', '-l1', type=str, nargs='*', default=None, help='tracker1 line1 protocol1')
    parser.add_argument('--line2', '-l2', type=str, nargs='*', default=None, help='tracker2 line2 protocol2')

def compare_lines(args):
    if args.name is None:
        raise ValueError('Please provide an experiment name')
    if args.line1 is None:
        raise ValueError('Please provide a reference for the first line')
    if args.line2 is None:
        raise ValueError('Please provide a reference for the second line')
    
    from umap.parametric_umap import load_ParametricUMAP

    directory = os.path.dirname(os.path.abspath(__file__))

    os.chdir('/home/alexandre/workspace/maggotuba_scale_20')

    t1, l1, p1 = args.line1
    t2, l2, p2 = args.line2

    # get training log
    with open('config.json', 'r') as f:
        config = json.load(f)
        log_dir = config['log_dir']

    # load lines
    embeds1 = np.load(os.path.join(log_dir, args.name, 'embeddings', t1, l1, p1, 'encoded_trajs.npy'))
    embeds2 = np.load(os.path.join(log_dir, args.name, 'embeddings', t2, l2, p2, 'encoded_trajs.npy'))

    # Load the umapper
    umapper = load_ParametricUMAP(os.path.join(config['log_dir'], args.name, 'umap', 'parametric_umap'))

    # compute 2d embeddings
    embeds1_2d = umapper.transform(embeds1)
    embeds2_2d = umapper.transform(embeds2)

    kernel_size = 1.0
    xmin = np.min(np.concatenate([embeds1_2d[:,0], embeds2_2d[:,0]]))-1
    ymin = np.min(np.concatenate([embeds1_2d[:,1], embeds2_2d[:,1]]))-1
    xmax = np.max(np.concatenate([embeds1_2d[:,0], embeds2_2d[:,0]]))+1
    ymax = np.max(np.concatenate([embeds1_2d[:,1], embeds2_2d[:,1]]))+1

    xmin = min(xmin, ymin)
    ymin = xmin
    xmax = max(xmax, ymax)
    ymax = xmax
    
    # Create mesh
    span_x = np.linspace(xmin, xmax, 200)
    span_y = np.linspace(ymin, ymax, 200)
    xx, yy = np.meshgrid(span_x, span_y)

    kde1 = gaussian_kde(embeds1_2d, xx, yy)
    kde2 = gaussian_kde(embeds2_2d, xx, yy)
    max_kde = np.max(np.concatenate((kde1, kde2)))

    wf = computeWitnessFunctionOnMesh(xx, yy, embeds1_2d, embeds2_2d, kernel_size)
    wf_m = np.max(np.abs(wf))

    for i in range(1,4):
        plt.figure(figsize=(4,4))
        if i == 1:
            plt.imshow(kde1, extent=[xmin, xmax, ymin, ymax], origin='lower', cmap='Blues', interpolation='antialiased', vmin=   0., vmax=max_kde)
        elif i == 2:
            plt.imshow(kde2, extent=[xmin, xmax, ymin, ymax], origin='lower', cmap= 'Reds', interpolation='antialiased', vmin=   0., vmax=max_kde)
        elif i == 3:
            plt.imshow(  wf, extent=[xmin, xmax, ymin, ymax], origin='lower', cmap= 'RdBu', interpolation='antialiased', vmin=-wf_m, vmax=wf_m)

        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
        plt.gca().set_aspect('equal', 'box')
        plt.colorbar(shrink=0.7)
        # axs[0].set_xticks([])    
        # axs[0].set_yticks([])

        plt.tight_layout()
        plt.savefig(os.path.join(directory, f'figure_4B{i}.jpg'))

def gaussian_kde(data, xx, yy, bandwidth=1.):
    mesh_points = np.vstack([xx.flatten(), yy.flatten()])
    mesh_points = mesh_points.reshape(1, *mesh_points.shape)
    data = data.reshape(*data.shape, 1)

    N = np.sqrt(2.*np.pi*bandwidth**2)
    D2 = np.sum((mesh_points-data)**2, axis=1)

    return 1./N * np.mean(np.exp(-0.5*D2/bandwidth**2), axis=0).reshape(xx.shape)

if __name__ == '__main__':

    class args:
        pass

    args.name = 'experiment_1'
    args.line1 = ('t5', 'GMR_10A11_AE_01@UAS_TNT_2_0003', 'p_8_45s1x30s0s#p_8_105s10x2s10s#n#n@100')
    args.line2 = ('t5', 'FCF_attP2_1500062@UAS_TNT_2_0003', 'p_8_45s1x30s0s#p_8_105s10x2s10s#n#n@100')
    compare_lines(args)
