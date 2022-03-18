import sys, re
import numpy as np
import pandas as pd
from os.path import isdir, dirname, basename, splitext
from shutil import rmtree
from pathlib import Path
from db_building.AbfData import AbfData
from db_building.CoaExampleDb import ExampleDb

__location__ = dirname(Path(__file__).resolve())
sys.path.extend([__location__, f'{__location__}/..'])

from helper_functions import parse_output_path, parse_input_path


def main(args):
    out_path = parse_output_path(args.db_dir)
    if isdir(out_path):
        rmtree(out_path)
    if args.read_index:
        read_index_df = pd.read_csv(args.read_index, index_col=0)
        if args.db_type == 'train':
            file_list = list(read_index_df.query(f'fold').fn)
        else:  # test
            file_list = list(read_index_df.query(f'fold == False').fn)
    else:
        file_list = parse_input_path(args.abf_in, pattern='*.abf')
    db_name = out_path+'db.fs'
    error_fn = out_path+'failed_reads.txt'
    npz_path = out_path + 'test_squiggles/'
    npz_path = parse_output_path(npz_path)

    db = ExampleDb(db_name=db_name, target=args.target, width=args.width)
    nb_example_reads = 0

    # split out abf files in target and non-target
    target_idx = np.argwhere([args.target in basename(fl) for fl in file_list])[:,0]
    if not len(target_idx):
        raise ValueError('None of the abf files seem of target source (must have target in their file name)')

    # --- process abf files ---
    for i, file in enumerate(file_list):
        try:
            non_target_name = None if i in target_idx else re.search('cOA[0-9]+', basename(file)).group(0)  # todo: counts on specific naming of files!
            tr = AbfData(abf_fn=file, normalization=args.normalization, lowpass_freq=80)
            nb_pos = db.add_training_read(training_read=tr, non_target=non_target_name)
            if nb_example_reads < args.nb_example_reads and nb_pos > 0:
                event_name = non_target_name if non_target_name is not None else args.target
                tr_labels = np.repeat(np.array('bg', dtype=f'<U{len(event_name)}'), len(tr.raw))
                tr_labels[tr.flat_pos_indices] = event_name
                np.savez(npz_path + splitext(basename(file))[0], base_labels=tr_labels, raw=tr.raw)
            db.pack_db()
            if db.nb_pos > args.max_nb_examples:
                print('Max number of examples reached')
                break
        except (KeyError, ValueError) as e:
            with open(error_fn, 'a') as efn:
                efn.write('{fn}\t{err}\n'.format(err=e, fn=basename(file)))
            continue

    db.pack_db()
    if db.nb_pos == 0:
        raise ValueError(f'No positive examples found for target {args.target}')
