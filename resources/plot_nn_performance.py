import argparse
import os
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
sys.path.append(f'{__location__}/..')
from helper_functions import parse_input_path

def main():
    parser = argparse.ArgumentParser(description='Plot performance measures for a directory of CNNs')
    parser.add_argument('--nn-dir', type=str, required=True,
                        help='nns directory produced by run_production_pipeline')
    parser.add_argument('--svg', type=str, required=True)
    args = parser.parse_args()

    fn_list = parse_input_path(args.nn_dir, pattern='*performance.pkl')

    performance_dict = {}
    for fn in fn_list:
        with open(fn, 'rb') as fh:
            cur_dict = pickle.load(fh)
        filename_path = Path(fn)
        coa_target = filename_path.parts[-2]
        performance_dict[coa_target] = {'accuracy': cur_dict['val_binary_accuracy'][-1],
                                'precision': cur_dict['val_precision'][-1],
                                'recall': cur_dict['val_recall'][-1]}
    performance_df = pd.DataFrame.from_dict(performance_dict).T
    performance_df.to_csv(os.path.splitext(args.svg)[0] + '.csv')

    fig, (ax_pr, ax_acc) = plt.subplots(1,2, figsize=(8.25, 2.9375))

    # precision recall
    sns.scatterplot(x='recall', y='precision', data=performance_df, ax=ax_pr)
    for coa_target, tup in performance_df.iterrows():
        ax_pr.text(x=tup.recall, y=tup.precision, s=coa_target, fontsize=5)

    # accuracy
    sns.violinplot(y='accuracy', data=performance_df, color="0.8", ax=ax_acc)
    sns.stripplot(y='accuracy', data=performance_df, ax=ax_acc)
    ll = min(performance_df.precision.min(), performance_df.recall.min()) - 0.01
    ax_pr.set_ylim(ll,1); ax_pr.set_xlim(ll,1)
    ax_pr.set_aspect('equal')

    plt.tight_layout()
    # plt.show()
    fig.savefig(args.svg, dpi=400)
    plt.close()


if __name__ == '__main__':
    main()
