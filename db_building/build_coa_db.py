import sys, re
import numpy as np
from os.path import isdir, dirname
from shutil import rmtree
from pathlib import Path
import pandas as pd
from db_building.AbfData import AbfData
from db_building.CoaExampleDb import ExampleDb
__location__ = dirname(Path(__file__).resolve())
sys.path.extend([__location__, f'{__location__}/..'])

from resources.helper_functions import parse_output_path, parse_input_path


def main(args):
    out_path = parse_output_path(args.db_dir)
    if isdir(out_path):
        rmtree(out_path)
        Path(out_path).mkdir()

    file_list = parse_input_path(args.abf_in, pattern='*.abf')
    db_name = out_path+'db.fs'

    db = ExampleDb(db_name=db_name)
    unfiltered = True
    # --- process abf files ---
    for i, file in enumerate(file_list):
        print(f'Processing {file}')
        tr = AbfData(abf_fn=file, normalization=args.normalization,
                     lowpass_freq=80, baseline_fraction=0.65)
        # print(f'Event summaries for {tr.coa_type}')
        # print(pd.Series(tr.get_event_lengths()).describe())

        db.add_training_read(training_read=tr, unfiltered=unfiltered)
        db.pack_db()
        if db.nb_pos > args.max_nb_examples:
            print('Max number of examples reached')
            break
    db.pack_db()
