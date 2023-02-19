import sys, os, re
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

dir = sys.argv[1]
if dir[-1] != '/': dir += '/'

fig, bp_axes = plt.subplots(1,3, figsize=(10,4), sharey=True)

fn_list = [dir + fn for fn in os.listdir(dir)]
bp_fn_list = [fn for fn in fn_list if os.path.basename(fn).startswith('barplot')]
bpdf_list = []

def ci_fun(x):
    sf = x.std() / np.sqrt(len(x)) * 1.96
    return x.mean() - sf, x.mean() + sf

for bi, bp_fn in enumerate(bp_fn_list):
    bpdf = pd.read_csv(bp_fn, index_col=0)
    bpdf.loc[:, 'percentage'] = bpdf.frac * 100
    bpdf_list.append(bpdf)
    sns.barplot(y='frac', x='coa_type', hue='value_type',
                data=bpdf, ax=bp_axes[bi],
                errwidth=1.5, capsize=0.1,
                errorbar=ci_fun,
                # palette='Blues',
                # palette={'estimate': 'grey', 'ref': 'white'}
                )
    bp_axes[bi].get_legend().remove()

tick_step = 0.25
all_bpdf = pd.concat(bpdf_list).reset_index(drop=True)
frac_max = all_bpdf.frac.max()
frac_ax_max = (frac_max // tick_step + 1) * tick_step
frac_ticks = np.arange(0, frac_ax_max + 0.01 * tick_step, tick_step)
plt.yticks(frac_ticks)
plt.ylim([0,frac_ax_max])
#
# for bp_ax in bp_axes:
#     bp_ax.set_ylim(frac_ax_max)
#     bp_ax.set_yticks(frac_ticks)
#     bp_ax.invert_yaxis()
fig.savefig(dir + 'compound_figure_bps.svg', dpi=400)

# --- heatmap ---
fig, hm_ax = plt.subplots(1,1, figsize=(4,4))
mean_df = pd.read_csv(dir + 'confmats_mean.csv', index_col=0)
sd_df = pd.read_csv(dir + 'confmats_std.csv', index_col=0)

annot_mat = (mean_df.round(2).astype(str) + '\n±' + sd_df.round(2).astype(str))
sns.heatmap(data=mean_df, annot=annot_mat, fmt='s', cmap='Blues', ax=hm_ax,
            cbar_kws={'ticks': np.arange(0,1.1,0.25)})
hm_ax.set_aspect('equal', adjustable='box')

plt.tight_layout()
# hm_ax.legend(title='frac')

fig.savefig(dir + 'compound_figure_heatmap.svg', dpi=400)