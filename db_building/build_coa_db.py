import sys, yaml
from os.path import isdir, dirname
from pathlib import Path
from shutil import rmtree, copyfile

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
    copyfile(args.event_types, f'{out_path}event_types.yaml')

    with open(args.event_types, 'r') as fh:
        event_type_dict = yaml.load(fh, yaml.FullLoader)

    db = ExampleDb(db_name=db_name)
    # By default, only use unfiltered segments
    unfiltered = False
    # --- process abf files ---
    for i, file in enumerate(file_list):
        print(f'Processing {file}')
        tr = AbfData(abf_fn=file, normalization=args.normalization,
                     lowpass_freq=80, baseline_fraction=0.65, event_type_dict=event_type_dict)
        print(f'Event summaries for {tr.coa_type}')
        print(pd.Series(tr.get_event_lengths()).describe())

        db.add_training_read(training_read=tr, unfiltered=unfiltered)
        db.pack_db()
        if db.nb_pos > args.max_nb_examples:
            print('Max number of examples reached')
            break
        print(f'total # examples: {db.nb_pos}')
    db.pack_db()
    print(f'Done! DB contains {db.nb_pos} examples')
