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
tf.config.set_visible_devices(gpus[1:], 'GPU')

class CoaInference:
    """Class that uses generated CNNs to detect cOAs in abf files

    :param nn_dir: Path to directory that contains folders with cOA?/nn.h5 files
    :type nn_dir: Path
    """
    def __init__(self, nn):
        self.nn = tf.keras.models.load_model(nn)
        self.input_length = self.nn.input_shape[1]

    def predict_from_file(self, abf_path):
        """Counts cOAs in single input file

        :param abf_path: path to .abf file
        :type abf_path: str
        :return: Counter object that contains
        :rtype: Counter
        """
        abf = AbfData(abf_path)
        events = abf.get_pos(unfiltered=True)
        x_pad = np.expand_dims(pad_sequences(events, maxlen=self.input_length,
                                             padding='post', truncating='post',
                                             dtype='float32'), -1)
        y_hat = self.nn.predict(x_pad)
        y_hat = np.argmax(y_hat, axis=1)
        return [INDEX_TO_TARGET[i] for i in y_hat]

def main(args):
    inference_model = CoaInference(args.nn_path)

    y_true = []
    y_pred_list = []

    for i, abf in enumerate(Path(args.abf_in).iterdir()):
        # # Uncomment to omit the coa5 file
        # if 'cOA5' in abf.name:
        #     continue
        print(f'Processing {abf.name}')
        y_pred = inference_model.predict_from_file(str(abf))
        # print(y_pred)
        true_coa = abf.name[:4]
        y_true.extend([true_coa] * len(y_pred))
        y_pred_list.extend(y_pred)
        if i > 5:
            break

    print(confusion_matrix(y_true, y_pred_list))
    print('Balanced accuracy', balanced_accuracy_score(y_true, y_pred_list))

