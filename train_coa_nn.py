import importlib
import os
import sys

import numpy as np
import tensorflow as tf
import yaml
from sklearn.metrics import balanced_accuracy_score

from db_building.CoaExampleDb import ExampleDb

sys.path.append(f'{os.path.dirname(__file__)}/..')
from resources.helper_functions import parse_output_path

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.set_visible_devices(gpus[1:], 'GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


def load_db(db_dir, read_only=False):
    """Load database from given directory

    :param db_dir: path to directory, must contain a 'db.fs' file
    :type db_dir: str
    :param read_only: If database should be read only or not
    :type read_only: bool
    :return: database and squiggles
    """
    if db_dir[-1] != '/':
        db_dir += '/'
    db = ExampleDb(db_name=db_dir + 'db.fs', read_only=read_only)
    return db


def train(parameter_file, training_data, test_data, model_weights=None,
          quiet=False):
    # Load parameter file
    if type(parameter_file) == str:
        with open(parameter_file, 'r') as pf: params = yaml.load(pf, Loader=yaml.FullLoader)
    elif type(parameter_file) == dict:
        params = parameter_file
    else:
        raise ValueError(f'{type(parameter_file)} is not a valid data type for a parameter file')

    # Load train & test data
    test_db = load_db(test_data, read_only=True)
    train_db = load_db(training_data, read_only=True)

    # Create save-file for model if required
    cp_callback = None

    # create nn
    nn_class = importlib.import_module(f'nns.{params["nn_class"]}').NeuralNetwork
    nn = nn_class(**params, weights=model_weights,
                  cp_callback=cp_callback)

    # Start training
    x_val, y_val = test_db.get_training_set()

    x_train, y_train = train_db.get_training_set(oversampling=params['oversampling'])
    nn.train(x_train, y_train, x_val, y_val, quiet=quiet, epochs=params['epochs'])

    # Uncomment to print confusion matrix
    # Rows are true labels, columns are predicted labels
    prediction = nn.predict(x_val)
    true_labels = [int(np.where(i == 1)[0]) for i in y_val]
    print(tf.math.confusion_matrix(true_labels, prediction))
    print('Balanced accuracy', balanced_accuracy_score(true_labels, prediction))
    return nn


def main(args):
    nn_target_dir = parse_output_path(f'{args.nn_dir}')
    nn = train(args.parameter_file, args.training_db, args.test_db,
               args.model_weights, False)
    nn.model.save(f'{nn_target_dir}nn.h5')
