import numpy as np
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import yaml
import pandas as pd

def parse_summary_yaml(file):
    with open(file) as f:
        summary = yaml.safe_load(f)
    summary.pop('balanced_accuracy')
    return summary


def main(bootstrapped_out_dir, coa_abundances):
    master_dict = {}
    for file in bootstrapped_out_dir.glob('**/bs_?/summary_stats.yaml'):
        summary = parse_summary_yaml(file)
        total = sum(summary.values())
        for key, value in summary.items():
            if key not in master_dict:
                master_dict[key] = [value/total]
            else:
                master_dict[key].append(value/total)

    # print(master_dict)
    df = pd.DataFrame.from_dict(master_dict)
    df = pd.melt(df, value_vars=['cOA3', 'cOA4', 'cOA5', 'cOA6'])
    df.columns = ['coa type', 'count']
    df['actual'] = False

    gt_df = pd.DataFrame.from_dict({"coa type": ['cOA3', 'cOA4', 'cOA5', 'cOA6'],
                                    # "count": [0.25]*4,
                                    # "count": [0.1] + [0.7] + [0.1]*2,
                                    'count': [float(i) for i in coa_abundances],
                                    'actual': [True]*4})
    df = pd.concat([df, gt_df])
    sns.barplot(x='coa type', y='count', hue='actual', data=df)
    plt.savefig(bootstrapped_out_dir / 'abundances.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bs-path', type=Path, help='Path to bootstrapped_results/ folder', required=True)
    parser.add_argument('--coa-abundances', nargs='+', required=True, help='True relative abundances for cOA3, 4, 5, and 6. Seperated by a space. E.g. 0.1 0.7 0.1 0.1 ')
    args = parser.parse_args()
    main(args.bs_path, args.coa_abundances)


















