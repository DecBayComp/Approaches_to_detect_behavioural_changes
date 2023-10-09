import argparse

# setup functions
import os
import json
from maggotuba.behavior_model.args import default_config

class setup:
    def add_arguments(parser):
        parser.add_argument('data_folder', type=str)
        parser.add_argument('project_folder', type=str)
        parser.add_argument('--len_traj', default=20, type=int)
        parser.add_argument('--len_pred', default=None, type=int)
    
    def main(args):
        data_folder = os.path.abspath(args.data_folder)
        project_folder = os.path.abspath(args.project_folder)
        if args.len_pred is None:
            args.len_pred = args.len_traj

        os.mkdir(project_folder)
        config = default_config(project_folder, data_folder, args.len_traj, args.len_pred)
        with open(config['config'], 'w') as config_file:
            json.dump(config, config_file, indent=2)
        os.mkdir(config['log_dir'])



# database functions
import maggotuba.behavior_model.data.build_database_histogram as db_count
import maggotuba.behavior_model.data.build_sample_database as db_build
class db:
    def counts(args):
        db_count.main(args)

    def build(args):
        db_build.main(args)

    def outlines(args):
        db_build.main_outlines(args)

# model functions
import maggotuba.cli.cli_model as model
import maggotuba.cli.cli_model_clustering as model_clustering

# plot functions
import maggotuba.cli.cli_plot as plot

# mmd functions
import maggotuba.cli.cli_mmd as mmd

def main():
    parser = argparse.ArgumentParser(prog='maggotuba')
    subparsers = parser.add_subparsers()

    # Project setup interface
    parser_setup = subparsers.add_parser('setup')
    setup.add_arguments(parser_setup)
    parser_setup.set_defaults(func=setup.main)

    # Database creation interface
    parser_db = subparsers.add_parser('db')
    parser_db.set_defaults(func=lambda args: parser_db.print_usage())

    subparsers_db = parser_db.add_subparsers()
    parser_db_counts = subparsers_db.add_parser('count')
    db_count.add_arguments(parser_db_counts)
    parser_db_counts.set_defaults(func=db.counts)

    parser_db_build = subparsers_db.add_parser('build')
    db_build.add_arguments(parser_db_build)
    parser_db_build.set_defaults(func=db.build)

    parser_db_outlines = subparsers_db.add_parser('outlines')
    db_build.add_arguments_outlines(parser_db_outlines)
    parser_db_outlines.set_defaults(func=db.outlines)


    # Model interface
    parser_model = subparsers.add_parser('model')
    parser_model.set_defaults(func=lambda args: parser_model.print_usage())

    subparsers_model = parser_model.add_subparsers()
    parser_model_train = subparsers_model.add_parser('train')
    model.add_arguments_train(parser_model_train)
    parser_model_train.set_defaults(func=model.train)

    parser_model_eval = subparsers_model.add_parser('eval')
    model.add_arguments_eval(parser_model_eval)
    parser_model_eval.set_defaults(func=model.eval)

    parser_model_cluster = subparsers_model.add_parser('cluster')
    model_clustering.add_arguments(parser_model_cluster)
    parser_model_cluster.set_defaults(func=model_clustering.main)

    parser_model_embed = subparsers_model.add_parser('embed')
    model.add_arguments_embed(parser_model_embed)
    parser_model_embed.set_defaults(func=model.embed)

    # Plotting interface
    parser_plot = subparsers.add_parser('plot')
    parser_plot.set_defaults(func=lambda _: parser_plot.print_usage())

    subparsers_plot = parser_plot.add_subparsers()
    parser_plot_line_density = subparsers_plot.add_parser('line_density')
    plot.add_arguments_plot_line_density(parser_plot_line_density)
    parser_plot_line_density.set_defaults(func=plot.plot_line_density)

    parser_compare_lines = subparsers_plot.add_parser('compare_lines')
    plot.add_arguments_compare_lines(parser_compare_lines)
    parser_compare_lines.set_defaults(func=plot.compare_lines)

    parser_plot_trajectories = subparsers_plot.add_parser('plot_trajectories')
    plot.add_arguments_plot_trajectories(parser_plot_trajectories)
    parser_plot_trajectories.set_defaults(func=plot.plot_trajectories)

    # MMD interface
    parser_mmd = subparsers.add_parser('mmd')
    parser_mmd.set_defaults(func=lambda args: parser_mmd.print_usage())

    subparsers_mmd = parser_mmd.add_subparsers()
    parser_mmd_cmpmat = subparsers_mmd.add_parser('compute_mat')
    mmd.add_arguments_cmpmat(parser_mmd_cmpmat)
    parser_mmd_cmpmat.set_defaults(func=mmd.compute_matrix)

    parser_mmd_plotmatrix = subparsers_mmd.add_parser('plot_matrix')
    mmd.add_arguments_plot_matrix(parser_mmd_plotmatrix)
    parser_mmd_plotmatrix.set_defaults(func=mmd.plot_matrix)

    parser_mmd_find_hits = subparsers_mmd.add_parser('find_hits_slurm')
    mmd.add_argument_find_hits(parser_mmd_find_hits)
    parser_mmd_find_hits.set_defaults(func=mmd.find_hits)


    args = parser.parse_args()
    args.func(args)