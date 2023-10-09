import json
import os
import torch

from maggotuba.behavior_model.models.model import Trainer
from maggotuba.behavior_model.models.neural_nets import Encoder, Decoder
from maggotuba.behavior_model.data.datarun import DataRun

EXPERIMENT = 'experiment_1'

def get_experiment():
    return EXPERIMENT

def set_experiment(new_experiment):
    global EXPERIMENT
    EXPERIMENT = new_experiment
    return EXPERIMENT

def get_training_log(experiment_dir=True):
    with open('config.json', 'r') as f:
        config = json.load(f)
        log_dir = config['log_dir']
    if experiment_dir:
        return os.path.join(log_dir, EXPERIMENT)

    return log_dir

def load_config(experiment=EXPERIMENT):    
    config_path = os.path.join(get_training_log(experiment_dir=True), 'config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return config

def load_trainer(experiment=EXPERIMENT):
    config = load_config(experiment)
    trainer = Trainer(**config)

    model_params = torch.load(os.path.join(config['log_dir'], 'best_validated_model.pt'))
    trainer.load_state_dict(model_params)

    return trainer

def load_encoder(experiment=EXPERIMENT):
    config = load_config(experiment)
    encoder = Encoder(**config)

    model_params = torch.load(os.path.join(config['log_dir'], 'best_validated_encoder.pt'))
    encoder.load_state_dict(model_params)

    return encoder    

def load_decoder(experiment=EXPERIMENT):
    config = load_config(experiment)
    target = config['target']
    len_traj, len_pred = config['len_traj'], config['len_pred']
    output_length = ('past' in target)*len_pred + ('present' in target)*len_traj + ('future' in target)*len_pred
    decoder = Decoder(output_length=output_length, **config)

    model_params = torch.load(os.path.join(config['log_dir'], 'best_validated_decoder.pt'))
    decoder.load_state_dict(model_params)

    return decoder

def load_paramatric_umap(experiment=EXPERIMENT):
    from umap.parametric_umap import load_ParametricUMAP
    exp_dir = get_training_log(experiment_dir=True)

    umap = load_ParametricUMAP(os.path.join(exp_dir, 'umap', 'parametric_umap'))

    return umap

def load_datarun(experiment=EXPERIMENT):
    config = load_config(experiment)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    datarun = DataRun(device=device, **config)
    return datarun

def load_sample_database():
    pass

def load_line_data():
    pass


