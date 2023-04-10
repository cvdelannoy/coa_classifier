import os, sys, re, yaml, argparse
from pathlib import Path
import numpy as np
from os.path import basename
__location__ = Path(__file__).parent.resolve()
sys.path.append(f'{__location__}/..')

from helper_functions import parse_input_path, parse_output_path
from db_building.AbfData import AbfData
from sklearn.cluster import KMeans
from scipy.ndimage import gaussian_filter

etd_fn = f'{__location__}/../coa_types/coa_types_4.yaml'  # todo make argument (?)
with open(etd_fn, 'r') as fh:
    event_type_dict = yaml.load(fh, yaml.FullLoader)


def get_coa_type(bn):
    return re.search('cOA[0-9]+', bn).group(0)

def get_id(bn):
    return re.search('M[0-9]+', bn).group(0)

def get_mix(bn):
    return re.search('[0-9]{2}-[0-9]{2}-[0-9]{2}-[0-9]{2}', bn).group(0)

def get_id_combination(bn):
    id_list = []
    if 'coa' in args.abf_identifiers:
        id_list.append(get_coa_type(bn))
    if 'mix' in args.abf_identifiers:
        id_list.append(get_mix(bn))
    if 'mid' in args.abf_identifiers:
        id_list.append(get_id(bn))
    return tuple(id_list)

def cut_out_event(sig):
    sig_filt = gaussian_filter(sig, 1)
    baseline = np.median(sig_filt)
    cutoff = baseline * 0.8
    event_ids = np.where(sig_filt < cutoff)[0]
    if not len(event_ids):
        return None
    # sig_reshape = sig.reshape(-1,1)
    # km = KMeans(2).fit(sig_reshape)
    # km_eid = np.argmin(km.cluster_centers_)
    # km_labels = km.predict(sig_reshape)
    # event_ids = np.argwhere(km_labels == km_eid)[:,0]

    step_list = np.diff(event_ids)
    cut_points = np.where(step_list > 1)[0] + 1
    if len(cut_points):
        event_list = np.split(event_ids, cut_points)
        event_ids = event_list[np.argmax([len(x) for x in event_list])]
    ev_start, ev_end = max(0, event_ids[0] - 15), min(event_ids[-1] + 15, len(sig))
    event = sig[ev_start:ev_end]
    return event

norm_methods_using_abfs = ['mad', 'shift', 'ir']

parser = argparse.ArgumentParser(description='Convert txt-format extracted events to npz format readable by package')
parser.add_argument('--in-path', type=str, required=True,
                    help='path to folders each containing all events for a single abf file')
parser.add_argument('--abf-path', type=str, required=False,
                    help='path to original abf files,to extract normalization parameters')
parser.add_argument('--out-dir', type=str, required=True,
                    help='Directory where output npz should be stored')
parser.add_argument('--norm-method', type=str, choices=['old', 'mad', 'ir', 'shift', 'none'], default=['old'])
parser.add_argument('--abf-identifiers', type=str, nargs='+', default=['coa', 'mid'], choices=['coa', 'mix', 'mid'],
                    help='Define how matching abf should be found: based on [coa] type, M-id ([mid]) or combinations')
args = parser.parse_args()

out_dir = parse_output_path(args.out_dir)
if args.in_path[-1] != '/': args.in_path += '/'
dir_list = [args.in_path + x for x in os.listdir(args.in_path)]

for abf_path in dir_list:
    print(f'Parsing {basename(abf_path)}...')
    out_fn = out_dir + os.path.basename(abf_path) + '.npz'  # check fn, no slash
    sig_dict = {}

    # Extract properties for normalization
    if args.norm_method in norm_methods_using_abfs:
        txt_bn = basename(args.in_path)
        id_tup = get_id_combination(txt_bn)
        abf_dict = {get_id_combination(basename(x)): x for x in parse_input_path(args.abf_path, pattern='*.abf')}
        abf_fn = abf_dict.get(id_tup, False)
        if not abf_fn: continue
        abf = AbfData(abf_fn, lowpass_freq=80, baseline_fraction=0.65, event_type_dict=event_type_dict)
        bl = np.concatenate(abf.get_baseline())
        med = np.median(bl)
        mad = np.median(np.abs(bl - med))

    # extract and normalize
    for fn in parse_input_path(abf_path, pattern='*.dat'):
        sig = np.loadtxt(fn, comments='//')
        sig = sig[~np.isnan(sig)]
        # sig = cut_out_event(sig)
        if sig is None: continue
        if args.norm_method == 'mad':
            sig = (sig - med) / mad
        elif args.norm_method == 'shift':
            sig = sig - med
        elif args.norm_method == 'old':
            sig = (sig - np.median(sig)) / np.std(sig)
        elif args.norm_method == 'ir':
            sig = sig / med
        event_nb = re.search('(?<=event_)[0-9]+', fn).group(0)
        sig_dict[event_nb] = sig

    if len(sig_dict):
        np.savez(out_fn, **sig_dict)
