from tqdm import tqdm
import torch
import numpy as np
import torch.nn as nn
import torch.onnx
import os

from maggotuba.behavior_model.models.neural_nets import AutoEncoder
from maggotuba.behavior_model.data.datarun import DataRun
from maggotuba.behavior_model.features.build_features import features_larva
from maggotuba.behavior_model.visualization.visualize import Visualizer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
OPTIM = {'adam': torch.optim.Adam, 'sgd': torch.optim.SGD}
LOSS = {'MSE': nn.MSELoss}

class Trainer(nn.Module):
    def __init__(self,
                activation: str,
                batch_size: int, 
                bias: bool,
                cluster_penalty: str,
                cluster_penalty_coef: float,
                data_dir: str,
                dec_filters: int,
                dec_kernel: int,
                dec_depth: int,
                dim_latent: int,
                dim_reduc: str,
                enc_filters: int,
                enc_kernel: int,
                enc_depth: int,
                grad_clip: float, 
                init: str,
                len_pred: int,
                len_traj: int,
                length_penalty_coef: float,
                log_dir: str,
                loss: str,
                lr: int,
                n_clusters: int,
                n_features: int,
                num_workers: int, 
                optim_iter: int,
                pseudo_epoch: int,
                optimizer: str,
                target: set,
                device=device,
                **kwargs,
                ):
        super().__init__()

        self.n_features = n_features
        self.len_traj = len_traj
        self.len_pred = len_pred
        self.dim_latent = dim_latent

        self.fitted = False       # flag tracking whether the 'fit' method was run

        self.autoencoder = AutoEncoder(activation=activation,
                                       bias=bias,
                                       cluster_penalty=cluster_penalty,
                                       dec_filters=dec_filters,
                                       dec_kernel=dec_kernel,
                                       dec_depth=dec_depth,
                                       dim_latent=dim_latent,
                                       enc_filters=enc_filters,
                                       enc_kernel=enc_kernel,
                                       enc_depth=enc_depth,
                                       init=init,
                                       len_pred=len_pred,
                                       len_traj=len_traj,
                                       loss=loss,
                                       n_clusters=n_clusters,
                                       n_features=n_features,
                                       target=target,
                                       optimizer=optimizer,
                                       device=device)

        self.device = device
        self.init = init
        self._iter = 0
        self.batch_size = batch_size
        self.loss = LOSS[loss]
        self.cluster_penalty = cluster_penalty
        self.cluster_penalty_coef = cluster_penalty_coef
        self.length_penalty_coef = length_penalty_coef
        self.lr = lr
        self.optim_iter = optim_iter
        self.optimizer = optimizer
        self.optim = OPTIM[self.optimizer](self.parameters(), self.lr)
        self.pseudo_epoch = pseudo_epoch
        self.grad_clip = grad_clip
        self.target = target
        self.n_clusters = n_clusters
        self.eval_loss = 10e10

        self.log_dir = log_dir
        self.stop_training = False
        self.data = DataRun(n_features=self.n_features, len_traj=self.len_traj, len_pred=self.len_pred, batch_size=self.batch_size, 
                            num_workers=num_workers, device=self.device, data_dir=data_dir)
        self.dim_reduc = dim_reduc
        if 'path_saved_embedder' in kwargs:
            self.visu = Visualizer(log_dir=self.log_dir, target=self.target, dim_reduc=self.dim_reduc, path_saved_embedder=kwargs['path_saved_embedder'])
        else:
            self.visu = Visualizer(log_dir=self.log_dir, target=self.target, dim_reduc=self.dim_reduc)
        torch.set_num_threads(10)
        self.to(self.device)

    
    def fit(self):
        # Initial dry run to check shape consistency
        print('Dry run through encoder 0 :')
        self.autoencoder.encoder.dry_run()
        print('Dry run through decoder 0 :')
        self.autoencoder.decoder.dry_run()

        self.autoencoder.to(device)
        self.train()

        for step in range(self.optim_iter):
            if self.stop_training:
                break

            # optimization step
            self._single_step(step, verbose=True)

            # validation guard clause
            if self._iter % self.pseudo_epoch and self._iter < self.optim_iter:
                continue

            # validation
            self.eval()
            eval_loss, eval_results = self.evaluate(final=(self._iter==self.optim_iter))
            self._checkpoint(eval_loss)

            if eval_loss > self.eval_loss:
                pass
                # self.stop_training = True
            self.train()

            # figure creation
            self.save_figs(eval_results, final=(self._iter==self.optim_iter))

        self.fitted = True


    def _single_step(self, step, verbose=False):
        batch = self.data.sample('train').to(device)
        losses = self.autoencoder.losses(batch)
        loss = losses['training_loss']

        self.optim.zero_grad()
        loss.backward()


        training_loss = losses['training_loss']
        nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)
        self.optim.step()

        with torch.no_grad():
            gn = sum([p.grad.pow(2).sum() for p in self.parameters() if p.grad is not None]).sqrt().item()

        self.visu.train_losses.append(training_loss.item())
        self.visu.train_penalties.append(0.0)

        if verbose:
            print(f'step: {step: d} |'
                  f' training_loss: {float(training_loss.item()): .3e} |'
                  f' penalty_loss: {float(0.0): .3e} |'
                  f' grad_norm: {float(gn): .3e} |'
                  f' validation_loss: {float(self.eval_loss): .3e} |'
                  f" lr: {self.optim.param_groups[0]['lr']:.3e}")

        self._iter += 1

    def _checkpoint(self, eval_loss):
        if self.eval_loss > eval_loss:
            self.eval_loss = eval_loss
            self.saved_iter = self._iter

            # onnx export
            dummy_input = torch.randn((10, 2, self.n_features//2, self.len_traj)).float().to(device)
            torch.onnx.export(self.autoencoder.encoder, dummy_input, 
                              os.path.join(self.log_dir, 'best_validated_encoder.onnx'),
                              opset_version=10, input_names =['input'], output_names=['output'],
                              dynamic_axes={'input':{0: 'batch_size'}, 'output':{0: 'batch_size'}})
            dummy_input = torch.randn((10, 10)).float().to(device)
            torch.onnx.export(self.autoencoder.decoder, dummy_input, 
                              os.path.join(self.log_dir, 'best_validated_decoder.onnx'),
                              opset_version=10, input_names =['input'], output_names=['output'],
                              dynamic_axes={'input':{0: 'batch_size'}, 'output':{0: 'batch_size'}})

            # full model export
            torch.save(self.state_dict(), os.path.join(self.log_dir,'best_validated_model.pt'))

            # encoder/decoder export
            torch.save(self.autoencoder.encoder.state_dict(), os.path.join(self.log_dir, 'best_validated_encoder.pt'))
            torch.save(self.autoencoder.decoder.state_dict(), os.path.join(self.log_dir, 'best_validated_decoder.pt'))

        torch.save(self.state_dict(), os.path.join(self.log_dir, 'checkpoint.pt'))


    def evaluate(self, N=None, kw='val', final=False):
        # default value for N is different depending on final.
        if N is None:
            if final:
                N = self.data.n_val_batches
            else:
                N = self.data.n_val_batches//2+1
        # No point in looking at more batches than there is.
        N = min(N, self.data.n_val_batches)
        
        eval_results = {}
        eval_results['embeds'] = torch.empty(N*self.batch_size, self.dim_latent)
        eval_results['labels'] = torch.empty(N*self.batch_size)
        eval_results['future_labels'] = torch.empty(N*self.batch_size)
        eval_results['presents'] = torch.empty(N*self.batch_size, 2, self.n_features//2, self.len_traj)
       
        eval_results['past_rmse'] = torch.empty(N*self.batch_size)
        eval_results['present_rmse'] = torch.empty(N*self.batch_size)
        eval_results['future_rmse'] = torch.empty(N*self.batch_size)
        
        eval_loss = 0
        for i in tqdm(range(N), desc='evaluating'):
            try:
                batch = self.data.sample(kw)
                assert(batch.present.shape[0] == self.batch_size)
            except:
                N = i
                break
            eval_results['presents'][i*self.batch_size:(i+1)*self.batch_size] = batch.present
            with torch.no_grad():
                embed = self.autoencoder.encoder(batch.present)
            eval_results['embeds'][i*self.batch_size:(i+1)*self.batch_size] = embed # TODO : Maybe change to list append and then stack to avoid slicing
            eval_results['labels'][i*self.batch_size:(i+1)*self.batch_size] = batch.label['present_label']
            eval_results['future_labels'][i*self.batch_size:(i+1)*self.batch_size] = batch.label['future_label']
            with torch.no_grad():
                long_scale = self.autoencoder.decoder(embed, long_scale=True)
                eval_loss += self.autoencoder.losses(batch)['training_loss'].item()

            eval_results['past_rmse'][i*self.batch_size:(i+1)*self.batch_size] = torch.sqrt(torch.mean((batch.past.reshape(self.batch_size, -1)-long_scale[...,:self.len_pred].reshape(self.batch_size,-1))**2, dim=1))
            eval_results['present_rmse'][i*self.batch_size:(i+1)*self.batch_size] = torch.sqrt(torch.mean((batch.present.reshape(self.batch_size, -1)-long_scale[...,self.len_pred:self.len_pred+self.len_traj].reshape(self.batch_size,-1))**2, dim=1))
            eval_results['future_rmse'][i*self.batch_size:(i+1)*self.batch_size] = torch.sqrt(torch.mean((batch.future.reshape(self.batch_size, -1)-long_scale[...,self.len_pred+self.len_traj:].reshape(self.batch_size,-1))**2, dim=1))

        if N:
            eval_results['features_larva'] = features_larva(eval_results['presents'][:N*self.batch_size])
            eval_loss /= N
            self.visu.eval_losses.append(eval_loss)
            ids = torch.randperm(self.batch_size)[:6]
            batch = batch.to(torch.device('cpu'))
            eval_results['example_reconstructions'] = {'past' : batch.past[ids],
                                                       'present' : batch.present[ids],
                                                       'future' : batch.future[ids],
                                                       'label' : batch.label['present_label'][ids],
                                                       'pred' : long_scale[ids]}

            for k, v in eval_results.items():
                if k != 'example_reconstructions':
                    eval_results[k] = v.detach().cpu().numpy()
                else:
                    for subk, subv in v.items():
                        v[subk] = subv.detach().cpu().numpy()
                    eval_results[k] = v

        return eval_loss, eval_results

    def save_figs(self, eval_results, save_to='training', fit_embedder=True, fit_pairwise_embedder=True, fit_single_embedder=True, path_saved_embedder=None, _3d=False, final=False):
        self.cpu()
        torch.cuda.empty_cache()
        if not os.path.isdir(os.path.join(self.log_dir,'visu',save_to)):
            os.mkdir(os.path.join(self.log_dir,'visu',save_to))
        self.visu._iter = self._iter
        if path_saved_embedder:
            assert fit_embedder == False
            self.visu.path_saved_embedder = path_saved_embedder
        #Plot larva and predictions
        self.visu.plot_trajectory(eval_results['example_reconstructions'], save_to)

        #Plot the embedding distribution in latent space
        embed = np.concatenate([eval_results['embeds'], eval_results['features_larva']],-1)
        self.visu.plot_embed(embed, eval_results['labels'], self.cluster_centers.detach() if hasattr(self, 'cluster_centers') else None, kw=save_to, _3d=_3d, fit_embedder=fit_embedder)

        if final:
            # pass
            # Plot pairwise and single behaviour UMAP
            self.visu._compute_mean_label_entropy(eval_results['embeds'], eval_results['labels'], kw=save_to)
            self.visu._compute_plot_pairwise_umap(eval_results['embeds'], eval_results['labels'], kw=save_to, fit=fit_pairwise_embedder)
            self.visu._compute_plot_single_behaviour_umap(eval_results['embeds'], eval_results['labels'], kw=save_to, fit=fit_single_embedder)
            
            # Plot reconstruction error and transition map
            rmse = (eval_results['past_rmse'], eval_results['present_rmse'], eval_results['future_rmse'])
            self.visu._reconstruction_error(eval_results['embeds'], eval_results['labels'], *rmse, kw=save_to)
            self.visu._transition_map(eval_results['embeds'], eval_results['labels'], eval_results['future_labels'], kw=save_to)
            self.visu._transition_map(eval_results['embeds'], eval_results['labels'], eval_results['future_labels'], kind='runbend', kw=save_to)
            self.visu._transition_proba(eval_results['embeds'], eval_results['labels'], eval_results['future_labels'], kw=save_to)
            # for i in range(2,3,6):
            #     self.visu.gudhi_clustering_single_behavior(embed[:,:10], label, fit=fit_embedder, to_cluster=i)

            self.visu.train_evaluation_classifier(eval_results['embeds'], eval_results['labels'], kw=save_to)

        # Plot losses
        self.visu.plot_losses(kw=save_to)
        self.visu.join()
        self.to(device)

