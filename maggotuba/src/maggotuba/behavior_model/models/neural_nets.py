import torch
import torch.nn as nn
import torch.nn.functional as F

ACTIVATION = {'relu': nn.ReLU(), 'elu': nn.ELU(), 'tanh': nn.Tanh(), 'sigmoid': nn.Sigmoid()}
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_same_padding(k):
    # helper function to enable onnx export
    if k%2:
        return k//2, k//2
    else:
        return k//2, k//2-1

class Encoder(nn.Module):
    def __init__(self, enc_depth=3, enc_filters=64, activation='relu', enc_kernel=3,
                bias=False, n_features=10, len_traj=10, dim_latent=2, **kwargs):
        super().__init__()
        self.n_features = n_features
        self.len_traj = len_traj
        self.activation = ACTIVATION[activation]
        self.bias = bias
        self.channels = [int(enc_filters)]*(enc_depth-1) if not isinstance(enc_filters, (list, tuple)) else enc_filters         # layer widths, i.e. number of kernels per layer
        self.kernel_sizes = [int(enc_kernel)]*(enc_depth-1) if not isinstance(enc_kernel, (list, tuple)) else enc_kernel     # kernel sizes
        self.dim_latent = dim_latent

        self.convolutional_blocks = self._build_convolutional_blocks()
        self.output_layers = self._build_output_layers()

    def forward(self, x):        
        convolutions = self.convolutional_blocks(x)
        flattened_convolutions = convolutions.flatten(start_dim=1)
        output = self.output_layers(flattened_convolutions)
        
        return output

    def dry_run(self, x=None):
        with torch.no_grad():
            if x is None:
                x = torch.ones((128, 2, self.n_features//2, self.len_traj), device=device)
            print('x : ', x.shape)
            conv = x
            for i, conv_layer in enumerate(self.convolutional_blocks):
                conv = conv_layer(conv)
                print('conv_{} : '.format(i), conv.shape)
            out = conv.flatten(start_dim=1)
            print('flattened_convolutions : ', out.shape)
            for i, out_layer in enumerate(self.output_layers):
                out = out_layer(out)
                print('out_{} : '.format(i), out.shape)
        return out

    def _build_convolutional_blocks(self):
        '''Build the convolutional blocks of the network'''
        
        conv_blocks = [self._build_conv_block(in_channel, out_channel, kernel_size)
                       for in_channel, out_channel, kernel_size 
                       in zip([2]+self.channels[:-1], self.channels, self.kernel_sizes)]
        return nn.Sequential(*sum(conv_blocks, []))

    def _build_conv_block(self, _in, _out, kernel_size, d=1):
        '''Build a single convolutional block comprised of a 1-2D convolution, an activation, and a 1-2D batchnorm. NB : no pooling.'''
        padding_bodyparts = get_same_padding(kernel_size[0])
        padding_time = get_same_padding(kernel_size[1])
        padding = (*padding_time, *padding_bodyparts)
        layers = [nn.ZeroPad2d(padding),
                  nn.Conv2d(_in, _out, kernel_size=kernel_size, bias=self.bias), # Use zero padding. This way, the length of the time series stays constant
                  self.activation,
                  nn.BatchNorm2d(_out, affine=True, track_running_stats=True)]
        return layers

    def _build_output_layers(self):
        '''Build final layer of the network comprised of a fully-connected layer and a batchnorm.'''
        flattened_dim = self.len_traj*self.n_features//2*self.channels[-1]
        layers = [nn.Linear(flattened_dim, self.dim_latent),
                  nn.BatchNorm1d(self.dim_latent, affine=True, track_running_stats=True)]
        return nn.Sequential(*layers)


class Decoder(nn.Module):
    '''Decoder class.
       Takes in a batch of latent vectors. These are then upsampled by a ully connected layer and reshaped into the final shape.
       The batch in its final shape is then run through multiple convolutions in order to give it an appropriate structure.
    '''
    def __init__(self, dec_depth=3, dec_filters=64, activation='relu', dec_kernel=3, n_features=10, 
                bias = None, dim_latent=2, output_length=30, **kwargs):
        super().__init__()
        self.n_features = n_features
        self.dim_latent = dim_latent
        self.bias = bias
        self.channels = [int(dec_filters)]*(dec_depth-1) if not isinstance(dec_filters, (list, tuple)) else dec_filters
        self.kernel_sizes = [int(dec_kernel)]*(dec_depth-1) if not isinstance(dec_kernel, (list, tuple)) else dec_kernel
        self.activation = ACTIVATION[activation] 
        self.output_length = output_length

        self.upsampling_layer = self._build_upsampling_layers()
        self.conv_blocks = self._build_conv_blocks()
        self.output_layers = self._build_final_layers()

    def forward(self, z, *args, **kwargs):
        # z is of shape (batch_size, latent_dims)

        upsampled = self.upsampling_layer(z)
        reshaped = upsampled.reshape(-1, 1, self.n_features//2, self.output_length)
        convolutions = self.conv_blocks(reshaped)
        output = self.output_layers(convolutions)
        return output

    def dry_run(self, z=None):
        with torch.no_grad():
            if z is None:
                z = torch.ones((128, self.dim_latent), device=device)
            print('z : ', z.shape)
            upsampled = self.upsampling_layer(z)
            print('upsampled : ', upsampled.shape)
            reshaped = upsampled.reshape(-1, 1, self.n_features//2, self.output_length)
            print('reshaped : ', reshaped.shape)
            convt=reshaped

            for i, convt_layer in enumerate(self.conv_blocks):
                convt = convt_layer(convt)
                print('convt_{} : '.format(i), convt.shape)
            out = convt
            for i, output_layer in enumerate(self.output_layers):
                out = output_layer(out)
                print('out_{} : '.format(i), out.shape)
        return out

    def _build_conv_blocks(self):
        '''Build the block of the network'''
        conv_blocks = [self._build_conv_block(in_channel, out_channel, kernel_size)
                       for in_channel, out_channel, kernel_size
                       in zip([1]+self.channels[:-1], self.channels, self.kernel_sizes)]
        return nn.Sequential(*sum(conv_blocks, []))

    def _build_conv_block(self, _in_channel, _out_channel, kernel_size):
        '''Build a single deconvolutional block comprised of a 1-2D transposed convolution, an activation, and a 1-2D batchnorm.'''
        padding_bodyparts = get_same_padding(kernel_size[0])
        padding_time = get_same_padding(kernel_size[1])
        padding = (*padding_time, *padding_bodyparts)
        layers = [nn.ZeroPad2d(padding),
                  nn.Conv2d(_in_channel, _out_channel, kernel_size=kernel_size, bias=self.bias),
                  self.activation,
                  nn.BatchNorm2d(_out_channel, affine=True, track_running_stats=True)]
        return layers

    def _build_final_layers(self):
        '''Build final layer of the network comprised of a fully-connected layer and a batchnorm.'''

        layers = [nn.Conv2d(self.channels[-1], 2, (1,1)),
                  nn.BatchNorm2d(2, affine=True, track_running_stats=True)]

        return nn.Sequential(*layers)

    def _build_upsampling_layers(self, mlp_depth=10):
        upsampled_dim = self.n_features//2*self.output_length
        return nn.Linear(self.dim_latent, upsampled_dim)

class AutoEncoder(nn.Module):
    '''Container for a Encoder and a Decoder.
       Responsible for computing the losses on a particular batch.
    '''

    LOSS = {'MSE': nn.MSELoss}

    def __init__(self,
                activation: str,
                bias: bool,
                cluster_penalty: str,
                dec_filters: int,
                dec_kernel: int,
                dec_depth: int,
                dim_latent: int,
                enc_filters: int,
                enc_kernel: int,
                enc_depth: int,
                init: str,
                len_pred: int,
                len_traj: int,
                loss: str,
                n_clusters: int,
                n_features: int,
                target: set,
                optimizer: str,
                device: str,
                **kwargs):
        super().__init__()
        self.n_features = n_features
        self.len_traj = len_traj
        self.len_pred = len_pred
        self.dim_latent = dim_latent
        self.device = device

        self.target = target
        self.output_length = ('past' in target)*len_pred + ('present' in target)*len_traj + ('future' in target)*len_pred

        self.encoder = Encoder(enc_depth=enc_depth, enc_filters=enc_filters, activation=activation, enc_kernel=enc_kernel,
                            bias=bias, n_features=n_features, len_traj=len_traj, dim_latent=dim_latent).float()
        self.decoder = Decoder(dec_depth=dec_depth, dec_filters=dec_filters, activation=activation, dec_kernel=dec_kernel,
                            bias=bias, n_features=n_features, output_length=self.output_length, dim_latent=dim_latent).float()

        self.init = init

        self.loss = AutoEncoder.LOSS[loss]()

        self.cluster_penalty = cluster_penalty
        self.n_clusters = n_clusters

        self.target = target
        self.apply(self._init_weights)
        self.optimizer = optimizer



    def _init_weights(self, m):
        if isinstance(m,(nn.Linear,nn.Conv2d,nn.Conv1d,nn.ConvTranspose2d,nn.ConvTranspose1d)):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if self.init == 'orthogonal':
                nn.init.orthogonal_(m.weight)
            if self.init == 'xavier':
                nn.init.xavier_uniform_(m.weight)
            if self.init == 'kaiming':
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')

    def losses(self, batch, to_compute={'training_loss'}):
        losses = {}

        embed = self.encoder(batch.present)
        pred = self.decoder(embed)
        target_list = ('past' in self.target)*[batch.past] + ('present' in self.target)*[batch.present] + ('future' in self.target)*[batch.future]
        target = torch.cat(target_list, -1)

        if 'training_loss' in to_compute:
            training_loss = torch.mean(self.loss(pred, target))
            losses['training_loss'] = training_loss
        
        return losses