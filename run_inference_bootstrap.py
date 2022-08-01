import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from snakemake import snakemake
from jinja2 import Template
from pathlib import Path
from glob import glob
import yaml

from resources.helper_functions import parse_output_path

__location__ = str(Path(__file__).resolve().parents[0])

def main(args):
    with open(args.event_types, 'r') as fh:
        event_type_dict = yaml.load(fh, yaml.FullLoader)
    TARGET_TO_INDEX = {x: i for i, x in enumerate(np.unique(list(event_type_dict.values())))}
    _ = parse_output_path(args.out_dir, clean=True)
    bs_dir = parse_output_path(args.out_dir + 'bootstrapped_results')
    log_dir = parse_output_path(args.out_dir + 'logs')


    with open(__location__ + '/run_inference_bootstrap.sf') as fh:
        sm_text = fh.read()
    sm_txt_out = Template(sm_text).render(
        __location__=__location__,
        bs_dir=bs_dir,
        log_dir=log_dir,
        nn_path=args.nn_path,
        abf_in=args.abf_in,
        bootstrap_iters=args.bootstrap_iters,
        event_types=str(Path(args.event_types).resolve())
    )
    sf_fn = args.out_dir + 'run_inference_bootstrap.sf'
    with open(sf_fn, 'w') as fh: fh.write(sm_txt_out)

    snakemake(snakefile=sf_fn, cores=args.cores, keepgoing=True, resources={'gpu': 1})

    # --- combine results ---
    analysis_dir = parse_output_path(args.out_dir + 'analysis')
    confmat_array = np.dstack([np.loadtxt(fn) for fn in glob(f'{bs_dir}*/confmat.csv')])
    labels_list = []
    for fn in glob(f'{bs_dir}*/confmat_labels.txt'):
        with open(fn, 'r') as fh: labels_list.append([ll.strip() for ll in fh.readlines()])
    assert np.all( [np.all(labels_list[0] == ll) for ll in labels_list])
    labels = labels_list[0]
    np.save(f'{analysis_dir}confmats.npy', confmat_array)

    if args.normalize_rates:
        with open(f'{__location__}/resources/coa_rates.yaml', 'r') as fh:
            rates_dict = yaml.load(fh, yaml.FullLoader)
        rates_vec = np.array([rates_dict.get(l, 1.0) for l in labels])
        rates_vec = np.expand_dims(np.expand_dims(rates_vec, 0), -1)
        confmat_array = confmat_array / rates_vec

    # row-normalize
    confmat_norm = (confmat_array.transpose((1, 0, 2)) / confmat_array.sum(axis=1)).T
    confmat_norm = np.nan_to_num(confmat_norm)
    np.save(f'{analysis_dir}confmats_normalized.npy', confmat_norm)

    # mean and std
    mean_df = pd.DataFrame(confmat_norm.mean(axis=0), index=labels, columns=labels)
    mean_df.rename_axis('Truth', axis='rows', inplace=True)
    mean_df.rename_axis('Predicted', axis='columns', inplace=True)
    mean_df.to_csv(f'{analysis_dir}confmats_mean.csv')
    sd_df = pd.DataFrame(confmat_norm.std(axis=0), index=labels, columns=labels)
    sd_df.to_csv(f'{analysis_dir}confmats_std.csv')

    # Collect summary_stats
    stats_list = []
    for fn in glob(f'{bs_dir}*/summary_stats.yaml'):
        with open(fn, 'r') as fh: stats_list.append(yaml.load(fh, yaml.FullLoader))
    stats_names = list(stats_list[0])
    nb_stats_files = len(stats_list)
    raw_tup_list = []
    tup_list = []
    for sn in stats_names:
        val_list = [sl[sn] for sl in stats_list]
        raw_tup_list.append(pd.Series({f'bs{vi}': v for vi, v in enumerate(val_list)}, name=sn))
        tup_list.append(pd.Series({'stat_mean': np.mean(val_list), 'stat_std': np.std(val_list)}, name=sn))
    stats_df = pd.concat(tup_list, axis=1).T
    raw_stats_df = pd.concat(raw_tup_list, axis=1).T
    stats_df.to_csv(f'{analysis_dir}stats_summary.csv')
    raw_stats_df.to_csv(f'{analysis_dir}raw_stats_summary.csv')

    # --- plot ---
    fig, ax = plt.subplots(figsize=(5, 5))
    annot_mat = (mean_df.round(2).astype(str) + '\n±' + sd_df.round(2).astype(str))
    sns.heatmap(data=mean_df, annot=annot_mat, fmt='s', cmap='Blues')
    ax.set_aspect('equal', adjustable='box')
    ax.legend(title='frac')
    plt.savefig(f'{analysis_dir}heatmap.svg', dpi=400)
    plt.close(fig)
