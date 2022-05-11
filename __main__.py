#!/usr/bin/python3
import argparse
import sys

import run_inference
import run_production_pipeline
import train_coa_nn
from db_building import build_coa_db
from resources import coa_argparse_dicts as argparse_dicts


def main(args=None):
    if args is None:
        args = sys.argv[1:]

    commands = [
        ('run_production_pipeline',
         'Generate DBs from read set and generate NNs for several cOAs at once',
         argparse_dicts.get_run_production_pipeline_parser(),
         run_production_pipeline.main),
        ('train_nn',
         'Train a single NN to detect a given cOA',
         argparse_dicts.get_training_parser(),
         train_coa_nn.main),
        ('build_db',
         'Build a training database, to train an NN for a given cOA',
         argparse_dicts.get_build_db_parser(),
         build_coa_db.main),
        ('run_inference',
         'Start up inference routine and classify occurences of cOA in abf file',
         argparse_dicts.get_run_inference_parser(),
         run_inference.main),
    ]

    parser = argparse.ArgumentParser(
        prog='baseLess',
        description='Build small neural networks to detect specific cOAs in ABF files.'
    )
    subparsers = parser.add_subparsers(
        title='commands'
    )

    for cmd, hlp, ap, fnc in commands:
        subparser = subparsers.add_parser(cmd, add_help=False, parents=[ap, ])
        subparser.set_defaults(func=fnc)
    args = parser.parse_args(args)
    args.func(args)


if __name__ == '__main__':
    main()
