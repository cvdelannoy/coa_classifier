import os, re, yaml, shutil
from os.path import basename
from random import shuffle
from jinja2 import Template
from snakemake import snakemake
import pandas as pd
import numpy as np
from run_inference import run_inference
from resources.run_inference_performance_analysis import run_inference_performance_analysis
from resources.helper_functions import parse_output_path, parse_input_path

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))


def main(args):
    shutil.rmtree(args.out_dir, ignore_errors=True)
    meta_db_dir = parse_output_path(f'{args.out_dir}dbs/')
    meta_nn_dir = parse_output_path(f'{args.out_dir}nns/')
    meta_logs_dir = parse_output_path(f'{args.out_dir}logs/')
    meta_abf_dir = parse_output_path(f'{args.out_dir}abfs/')
    meta_inference_dir = parse_output_path(f'{args.out_dir}inference/')
    with open(args.parameter_file, 'r') as pf: params = yaml.load(pf, Loader=yaml.FullLoader)
    with open(args.event_types, 'r') as fh: event_type_dict = yaml.load(fh, yaml.FullLoader)

    # Divide abfs up in folds
    abf_list = parse_input_path(args.abf_in, pattern='*.abf') + parse_input_path(args.abf_in, pattern='*.npz')
    abf_type_list = [event_type_dict[re.search('cOA[0-9]+', basename(fn)).group(0).lower()] for fn in abf_list]
    abf_df = pd.DataFrame({'fn': abf_list, 'coa_type': abf_type_list, 'fold': -1}).set_index('fn')
    nb_folds = min(args.nb_folds, min(np.unique(abf_type_list, return_counts=True)[1]))
    for coa_type, sdf in abf_df.groupby('coa_type'):
        idx_list = list(sdf.index)
        shuffle(idx_list)
        for ii, i in enumerate(idx_list):
            abf_df.loc[i, 'fold'] = ii % nb_folds

    for cv in range(args.nb_folds):
        db_dir = parse_output_path(f'{meta_db_dir}{cv}')
        nn_dir = parse_output_path(f'{meta_nn_dir}{cv}')
        logs_dir = parse_output_path(f'{meta_logs_dir}{cv}')
        abf_test_dir = parse_output_path(f'{meta_abf_dir}{cv}_test')
        abf_train_dir = parse_output_path(f'{meta_abf_dir}{cv}_train')
        inference_dir = parse_output_path(f'{meta_inference_dir}{cv}')

        # Define which abfs are part of training and test set in this fold
        for fn, tup in abf_df.iterrows():
            if tup.fold == cv:
                os.symlink(fn, abf_test_dir + basename(fn))
            else:
                os.symlink(fn, abf_train_dir + basename(fn))

        # classifier training: construct and run snakemake pipeline
        with open(f'{__location__}/run_production_pipeline.sf', 'r') as fh: template_txt = fh.read()
        sm_text = Template(template_txt).render(
            __location__=__location__,
            db_dir=db_dir,
            nn_dir=nn_dir,
            logs_dir=logs_dir,
            parameter_file=args.parameter_file,
            train_reads=abf_train_dir,
            test_reads=abf_test_dir,
            filter_width=params['filter_width'],
            event_types=args.event_types
        )

        sf_fn = f'{args.out_dir}nn_production_pipeline_{cv}.sf'
        with open(sf_fn, 'w') as fh: fh.write(sm_text)
        snakemake(sf_fn, cores=args.cores, verbose=False, keepgoing=True, resources={'gpu': 1})

        # Inference: run snakemake pipeline
        run_inference(abf_test_dir, f'{nn_dir}nn.h5', inference_dir, False, False)

    run_inference_performance_analysis(meta_inference_dir, f'{args.out_dir}analysis', True, False)

    cp=1