import argparse, os, sys

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from os.path import basename

from pathlib import Path
sys.path.append(f'{os.path.dirname(__file__)}/..')
from resources.helper_functions import parse_output_path

parser = argparse.ArgumentParser(description='Plot barplot of mixtures with reference bars')
parser.add_argument('--npy-in', type=str,required=True,
                    help='confmats_normalized.npy produced by run_inference_bootstrapped')
parser.add_argument('--class-names', type=str, required=True, nargs='+')
parser.add_argument('--ref-values', type=float, required=True, nargs='+')
parser.add_argument('--out-svg', type=str, required=True)
args = parser.parse_args()

conf_mat = np.load(args.npy_in)[:, -1, :-1]
nb_bs, nb_classes = conf_mat.shape
assert nb_classes == len(args.class_names)
assert nb_classes == len(args.ref_values)

est_df = pd.DataFrame(conf_mat.T, columns=[f'bs{i}' for i in range(nb_bs)], index=args.class_names)
est_df.reset_index(inplace=True)
est_df.rename({'index': 'coa_type'}, axis=1, inplace=True)
est_df = est_df.melt(id_vars=['coa_type'], var_name='bs_iter', value_name='frac')
est_df.loc[:, 'value_type'] = 'estimate'

ref_df = pd.DataFrame({'coa_type': args.class_names, 'frac': args.ref_values, 'bs_iter': 'bs0', 'value_type': 'ref'})
df = pd.concat((est_df, ref_df))

df.to_csv(f'{basename(args.out_svg)}.csv')
sns.barplot(y='frac', x='coa_type', hue='value_type', data=df, errorbar=('sd', 1.0))
plt.savefig(args.out_svg)
