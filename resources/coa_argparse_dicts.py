import argparse
import os, sys
from pathlib import Path

sys.path.append(f'{os.path.dirname(__file__)}/..')
from resources.helper_functions import parse_output_path


__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

# --- GENERAL ARGUMENTS ---
# --- inputs ---
abf_in = ('--abf-in', {
    'type': lambda x: check_input_path(x),
    'required': True,
    'help': 'Folder containing abf files'
})


training_abfs = ('--training-abfs', {
    'type': str,
    'required': True,
    'help': 'Directory containing abf format reads, for nn training.'
})

test_abfs = ('--test-abfs', {
    'type': str,
    'required': True,
    'help': 'Directory containing abf format reads, for nn testing.'
})

parameter_file = ('--parameter-file', {
    'type': str,
    'required': False,
    'default': os.path.join(__location__,
                            '../nns/hyperparams/CnnParameterFile_coa.yaml'),
    'help': 'a yaml-file containing NN parameters. If none supplied, default values are used.'})

# --- outputs ---
db_dir = ('--db-dir', {
    'type': str,
    'required': True,
    'help': 'Name (directory) new database'
})

target = ('--target', {
    'type': str,
    'required': True,
    'help': 'Target cOA'
})

out_dir = ('--out-dir', {
    'type': lambda x: parse_output_path(x),
    'required': True
})


# --- params ---
normalization = ('--normalization', {
    'type': str,
    'required': False,
    'default': 'median',
    'help': 'Specify how raw data should be normalized [default: median]'
})

silent = ('--silent', {
    'action': 'store_true',
    'help': 'Run without printing to console.'
})

cores = ('--cores', {
    'type': int,
    'default': 4,
    'help': 'Maximum number of CPU cores to engage at once.'
})


# --- PARSERS ---
def get_run_production_pipeline_parser():
    coa_list = ('--coa-list', {
        'type': str,
        'help': 'txt list of k-mers',
        'default': ''  # need it to be existent; passed to production pipeline
    })

    parser = argparse.ArgumentParser(description='Generate DBs from read sets and generate RNNs for several k-mers '
                                                 'at once')
    for arg in (training_abfs, test_abfs, out_dir, coa_list, cores,
                parameter_file):
        parser.add_argument(arg[0], **arg[1])
    return parser


def get_training_parser():
    training_db = ('--training-db', {
        'type': lambda x: check_db_input(x),
        'required': True,
        'help': 'Database generated by build_db, for training.'
    })

    test_db = ('--test-db', {
        'type': lambda x: check_db_input(x),
        'required': True,
        'help': 'Database generated by build_db, for testing.'
    })

    nn_dir = ('--nn-dir', {
        'type': lambda x: parse_output_path(x),
        'required': True,
        'help': 'Directory where produced network(s) are stored. Networks get target cOA as name.'
    })

    tensorboard_path = ('--tensorboard-path', {
        'type': str,
        'required': False,
        'default': os.path.expanduser('~/tensorboard_logs/'),
        'help': 'Define different location to store tensorboard files. Default is home/tensorboard_logs/. '
                'Folders and sub-folders are generated if not existiing.'
    })

    plots_path = ('--plots-path', {
        'type': str,
        'required': False,
        'default': None,
        'help': 'Define different location to store additional graphs, if made. Default is None (no graphs made) '
                'Folders and sub-folders are generated if not existiing.'
    })

    model_weights = ('--model-weights', {
        'type': str,
        'required': False,
        'default': None,
        'help': 'Provide a (tensorflow checkpoint) file containing graph meta data and weights '
                'for the selected model. '
    })

    ckpt_model = ('--ckpt-model', {
        'type': str,
        'required': False,
        'default': None,
        'help': 'Store the model weights at the provided location, with the same name as the parameter (yaml) file '
                '(with ckpt-extension).'
    })

    parser = argparse.ArgumentParser(description='Train a network to detect a given cOA in abf data.')
    for arg in (training_db, test_db, nn_dir, tensorboard_path, plots_path, parameter_file,
                model_weights, ckpt_model):
        parser.add_argument(arg[0], **arg[1])
    return parser


def get_build_db_parser():
    parser = argparse.ArgumentParser(description='Create ZODB database for target cOAs from ABF files')


    max_nb_examples = ('--max-nb-examples', {
        'type': int,
        'default': 10000,
        'help': 'Maximum number of examples to store in DB [default: 10000]'
    })

    for arg in (abf_in, db_dir, normalization, max_nb_examples):
        parser.add_argument(arg[0], **arg[1])
    return parser


def get_run_inference_parser():
    nn_dir = ('--nn-dir', {
        'type': Path,
        'required': True,
        'help': 'Directory where produced networks are stored.'
    })

    parser = argparse.ArgumentParser(description='Start up inference for abf files.')
    for arg in (abf_in, out_dir, nn_dir):
        parser.add_argument(arg[0], **arg[1])
    return parser


# --- argument checking ---
def check_db_input(db_fn):
    """
    Check existence and structure, remove db.fs extension if necessary
    """
    if db_fn.endswith('db.fs'):
        if not os.path.isfile(db_fn):
            raise_(f'Database {db_fn} does not exist')
        return db_fn[:-5]
    elif not os.path.isdir(db_fn):
        raise_(f'Database {db_fn} does not exist')
    if db_fn[-1] != '/': db_fn += '/'
    if not os.path.isfile(f'{db_fn}db.fs'):
        raise_(f'{db_fn} not recognized as usable database (no db.fs found)')
    return db_fn


def check_input_path(fn):
    """
    Check if path exists, add last / if necessary
    """
    if not os.path.isdir(fn):
        raise_(f'Directory {fn} does not exist')
    if fn[-1] != '/': fn += '/'
    return fn


def raise_(ex):
    """
    Required to raise exceptions inside a lambda function
    """
    raise Exception(ex)


if __name__ == '__main__':
    raise ValueError('argument parser file, do not call.')
