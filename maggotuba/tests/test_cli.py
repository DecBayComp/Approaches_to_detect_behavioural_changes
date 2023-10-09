import os
from argparse import ArgumentParser
import maggotuba.cli.cli as cli
import multiprocessing
import json

def test_integration_test(raw_data_dir, workspace, mock_pool, monkeypatch):
    # monkeypatching pool
    monkeypatch.setattr("multiprocessing.Pool", mock_pool)


    # setting up the project
    setup_parser = ArgumentParser()
    cli.setup.add_arguments(setup_parser)
    setup_args = setup_parser.parse_args([str(raw_data_dir), str(workspace / 'maggotuba')])

    cli.setup.main(setup_args)

    assert(os.path.exists(workspace / 'maggotuba'))

    # change to project folder
    os.chdir(workspace / 'maggotuba')

    # edit the config file to make the tests faster
    with open(workspace/'maggotuba'/'config.json', 'r') as f:
        config = json.load(f)
    config['enc_filters'] = [3,3]
    config['enc_kernels'] = [(2,2), (2,2)]
    config['enc_depth'] = 2
    config['dec_filters'] = [3,3]
    config['dec_kernels'] = [(2,2), (2,2)]
    config['dec_depth'] = 2
    config['optim_iter'] = 20
    config['pseudo_epoch'] = 10
    with open(workspace/'maggotuba'/'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    # count samples
    db_count_parser = ArgumentParser()
    cli.db_count.add_arguments(db_count_parser)
    db_count_args = db_count_parser.parse_args(['--n_workers', '1'])
    cli.db.counts(db_count_args)

    assert(os.path.exists(workspace / 'maggotuba' / 'counts.npy'))

    # create database
    db_build_parser = ArgumentParser()
    cli.db_build.add_arguments(db_build_parser)
    db_build_args = db_build_parser.parse_args(['--n_workers', '1', '--n_samples', '10000'])
    cli.db.build(db_build_args)

    assert(any([s.endswith('.hdf5') for s in os.listdir(workspace/'maggotuba')]))

    # train a model
    model_train_parser = ArgumentParser()
    cli.model.add_arguments_train(model_train_parser)
    model_train_args = model_train_parser.parse_args([])
    cli.model.train(model_train_args)

    assert(os.path.exists(workspace/'maggotuba'/'training_log'/'experiment_1'/'best_validated_encoder.onnx'))
    assert(os.path.exists(workspace/'maggotuba'/'training_log'/'experiment_1'/'best_validated_encoder.pt'))
    assert(os.path.exists(workspace/'maggotuba'/'training_log'/'experiment_1'/'best_validated_decoder.onnx'))
    assert(os.path.exists(workspace/'maggotuba'/'training_log'/'experiment_1'/'best_validated_decoder.pt'))
    assert(os.path.exists(workspace/'maggotuba'/'training_log'/'experiment_1'/'visu'/'training'/'larva_iter_10.gif'))

    # evaluate the model
    model_eval_parser = ArgumentParser()
    cli.model.add_arguments_eval(model_eval_parser)
    model_eval_args = model_eval_parser.parse_args(['--name', 'experiment_1'])
    cli.model.eval(model_eval_args)
    assert(os.path.exists(workspace/'maggotuba'/'training_log'/'experiment_1'/'visu'/'eval'/'larva_iter_0.gif'))

    # embed the complete database
    model_embed_parser = ArgumentParser()
    cli.model.add_arguments_embed(model_embed_parser)
    model_embed_args = model_embed_parser.parse_args(['--name', 'experiment_1'])
    cli.model.embed(model_embed_args)
    embeddings_root = workspace/'maggotuba'/'training_log'/'experiment_1'/'embeddings'
    assert(os.path.exists(embeddings_root/'t5'/'LINE_0'/'p_8_45s1x30s0s#p_8_105s10x2s10s#n#n@100'/'encoded_trajs.npy'))

    # compute the MMD matrix for one tracker
    mmd_cmpmat_parser = ArgumentParser()
    cli.mmd.add_arguments_cmpmat(mmd_cmpmat_parser)
    mmd_cmpmat_args = mmd_cmpmat_parser.parse_args(['--name', 'experiment_1', '--n_workers', '1', '--tracker', 't5'])
    cli.mmd.compute_matrix(mmd_cmpmat_args)
    mmd_root = workspace/'maggotuba'/'training_log'/'experiment_1'/'mmd'
    assert(os.path.exists(mmd_root/'t5_mmd.csv'))
    assert(os.path.exists(mmd_root/'t5_mmd.npz'))