import argparse, sys, yaml

from pathlib import Path
from glob import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(str(Path(__file__).resolve().parents[1]))
from resources.helper_functions import parse_output_path

__coa_classifier__ = str(Path(__file__).resolve().parents[1])


def run_inference_performance_analysis(inference_dir, analysis_dir, correct_rates, correct_bias):
    # --- combine results ---
    analysis_dir = parse_output_path(analysis_dir)
    confmat_array = np.dstack([np.loadtxt(fn) for fn in glob(f'{inference_dir}**/confmat.csv', recursive=True)])
    labels_list = []
    for fn in glob(f'{inference_dir}**/confmat_labels.txt', recursive=True):
        with open(fn, 'r') as fh: labels_list.append([ll.strip() for ll in fh.readlines()])
    assert np.all([np.all(labels_list[0] == ll) for ll in labels_list])
    labels = labels_list[0]
    labels_array = np.array(labels)
    np.save(f'{analysis_dir}confmats.npy', confmat_array)

    if correct_rates:
        with open(f'{__coa_classifier__}/resources/coa_rates.yaml', 'r') as fh:
            rates_dict = yaml.load(fh, yaml.FullLoader)
        rates_vec = np.array([rates_dict.get(l, 1.0) for l in labels])
        rates_vec = np.expand_dims(np.expand_dims(rates_vec, 0), -1)
        confmat_array = confmat_array / rates_vec

    # row-normalize
    confmat_norm = (confmat_array.transpose((1, 0, 2)) / confmat_array.sum(axis=1)).T
    confmat_norm = np.nan_to_num(confmat_norm)

    if correct_bias:  # note: only works for coa_types_3.yaml!!!
        theta = pd.read_csv(f'{__coa_classifier__}/resources/theta_3_v2.csv', index_col=0).to_numpy()
        for nar, ar in enumerate(confmat_norm.sum(axis=1)[:, :3]):
            confmat_norm[nar, -1, :3] = np.clip(np.matmul(np.linalg.inv(theta), ar), 0.0, 1.0)  # clipping is allowed according to Buonaccorsi p.26 (I think)

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
    for fn in glob(f'{inference_dir}**/summary_stats.yaml', recursive=True):
        with open(fn, 'r') as fh:
            tmp_dict = yaml.load(fh, yaml.FullLoader)
        ss_dict = {l: tmp_dict.get(l, 0) for l in labels}
        stats_list.append(ss_dict)
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parse and plot coa analysis data')
    parser.add_argument('--inference-dir', type=str, required=True)
    parser.add_argument('--analysis-dir', type=str, required=True)
    parser.add_argument('--correct-rates', action='store_true')
    parser.add_argument('--correct-bias',  action='store_true')
    args = parser.parse_args()
    run_inference_performance_analysis(args.inference_dir, args.analysis_dir, args.correct_rates, args.correct_bias)
