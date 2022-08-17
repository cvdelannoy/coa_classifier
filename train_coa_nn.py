import importlib, h5py
import os, pickle
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

def save_model(nn, event_type_dict, fn):
    assert fn.endswith('.h5')
    nn.model.save(fn)
    with h5py.File(fn, 'r+') as fh:
        fh.attrs['event_type_dict'] = str(event_type_dict)

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

    with open(f'{db_dir}event_types.yaml', 'r') as fh:
        event_type_dict = yaml.load(fh, yaml.FullLoader)
    db = ExampleDb(db_name=db_dir + 'db.fs', read_only=read_only, event_type_dict=event_type_dict)
    return db


def train(nn_target_dir, parameter_file, training_data, test_data, model_weights=None,
          quiet=False, pregen_test_set=None):
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
    assert train_db.event_type_dict == test_db.event_type_dict  # ensure that test and train dbs recognize same event types

    # Create save-file for model if required
    cp_callback = None

    # create nn
    nn_class = importlib.import_module(f'nns.{params["nn_class"]}').NeuralNetwork

    # create metric files
    metrics_fn = f'{nn_target_dir}performances.csv'
    with open(metrics_fn, 'w') as fh: fh.write('iteration\tbalanced_accuracy\n')

    # Start training
    # x_val, y_val = test_db.get_training_set()
    if pregen_test_set:
        with open(pregen_test_set, 'rb') as fh: test_list = pickle.load(fh)
        x_val, y_val = test_list[0]
    else:
        x_val, y_val = test_db.get_training_set()

    nn_list = []
    nn_acc_list = []
    for mr in range(params['restarts']):
        nn = nn_class(**params, weights=model_weights,
                      cp_callback=cp_callback, nb_classes=train_db.nb_targets)
        if os.path.exists(f'{nn_target_dir}nn_iter{mr}.h5'):
            nn.model = tf.keras.models.load_model(f'{nn_target_dir}nn_iter{mr}.h5')
        else:
            for i in range(params['redraws']):
                x_train, y_train = train_db.get_training_set(oversampling=params['oversampling'])
                nn.train(x_train, y_train, x_val, y_val, quiet=quiet, epochs=params['epochs'] // params['redraws'])
        tr_acc_list = []
        for i in range(10):
            x_train, y_train = train_db.get_training_set(oversampling=params['oversampling'])
            tr_acc_list.append(balanced_accuracy_score([int(np.where(i == 1)[0]) for i in y_train], nn.predict(x_train)))
        nn_list.append(nn); nn_acc_list.append(np.mean(tr_acc_list))
        val_acc = balanced_accuracy_score([int(np.where(i == 1)[0]) for i in y_val], nn.predict(x_val))
        with open(metrics_fn, 'a') as fh: fh.write(f'{mr}\t{val_acc}\n')
        save_model(nn, train_db.event_type_dict, f'{nn_target_dir}nn_iter{mr}.h5')
        print(f'Validation accuracy iter {mr}: {val_acc}')
        tf.keras.backend.clear_session()
    nn = nn_list[np.argmax(nn_acc_list)]

    # Uncomment to print confusion matrix
    # Rows are true labels, columns are predicted labels
    prediction = nn.predict(x_val)
    true_labels = [int(np.where(i == 1)[0]) for i in y_val]
    print(tf.math.confusion_matrix(true_labels, prediction))
    if pregen_test_set:
        acc_list = []
        for xv, yv in test_list: acc_list.append(balanced_accuracy_score([int(np.where(i == 1)[0]) for i in yv], nn.predict(xv)))
        print(f'Balanced accuracy over {len(test_list)} test draws: ', np.mean(acc_list))
    else:
        print('Balanced accuracy', balanced_accuracy_score(true_labels, prediction))
    return nn, train_db.event_type_dict


def main(args):
    nn_target_dir = parse_output_path(f'{args.nn_dir}')
    nn, event_type_dict = train(nn_target_dir, args.parameter_file, args.training_db, args.test_db,
               args.model_weights, False, args.pregen_test_set)
    save_model(nn, event_type_dict, f'{nn_target_dir}nn.h5')
