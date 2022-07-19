import os
from collections import Counter
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, balanced_accuracy_score
from tensorflow.keras.preprocessing.sequence import pad_sequences

from db_building.AbfData import AbfData

# Map target coa to correct index
TARGET_TO_INDEX = {'cOA3': 0,
                   'cOA4': 1,
                   'cOA5': 2,
                   'cOA6': 3}

INDEX_TO_TARGET = {v: k for k, v in TARGET_TO_INDEX.items()}

gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.set_visible_devices(gpus[1:], 'GPU')


class CoaInference:
    """Class that uses generated CNNs to detect cOAs in abf files

    :param nn_dir: Path to directory that contains folders with nn.h5 file
    :type nn_dir: Path
    """
    def __init__(self, nn):
        self.nn = tf.keras.models.load_model(nn)
        self.input_length = self.nn.input_shape[1]

    def predict_from_file(self, abf_path, bootstrap=False):
        """Counts cOAs in single input file

        :param abf_path: path to .abf file
        :type abf_path: str
        :param bootstrap: indicate if .abf file should be bootstrapped
        :type bootstrap: bool
        :return: Counter object that contains
        :rtype: Counter
        """
        abf = AbfData(abf_path)
        events = abf.get_pos(unfiltered=False)
        x_pad = np.expand_dims(pad_sequences(events, maxlen=self.input_length,
                                             padding='post', truncating='post',
                                             dtype='float32'), -1)
        if bootstrap:
            x_pad = x_pad[np.random.randint(0, len(x_pad), size=len(x_pad))]
        y_hat = self.nn.predict(x_pad)
        y_hat = np.argmax(y_hat, axis=1)
        return [INDEX_TO_TARGET[i] for i in y_hat]

def main(args):
    if args.no_gpu:
        # Limit resources
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
        os.environ["TF_NUM_INTEROP_THREADS"] = "1"
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.set_soft_device_placement(True)

    inference_model = CoaInference(args.nn_path)

    y_true = []
    y_pred_list = []

    for i, abf in enumerate(Path(args.abf_in).iterdir()):
        print(f'Processing {abf.name}')
        y_pred = inference_model.predict_from_file(abf, args.bootstrap)
        y_pred_list.extend(y_pred)
        true_coa = abf.name[:4]
        if not true_coa.lower().startswith('coa'):
            # Looking at file of unknown type
            true_coa = 'UNKNOWN'
        y_true.extend([true_coa] * len(y_pred))
        # print("Y pred", y_pred)
        # print('Y true', y_true)

    conf_mat = confusion_matrix(y_true, y_pred_list)
    np.savetxt(args.out_dir + 'confmat.csv', conf_mat)
    pred_counts = Counter(y_pred_list)

    with open(args.out_dir + 'summary_stats.yaml', 'w') as fh:
        fh.write(f'balanced_accuracy: {balanced_accuracy_score(y_true, y_pred_list)}\n')
        for key, value in pred_counts.items():
            fh.write(f"{key}: {value}\n")
    print(confusion_matrix(y_true, y_pred_list))
    print('Balanced accuracy', balanced_accuracy_score(y_true, y_pred_list))

