from pathlib import Path
import tensorflow as tf
import numpy as np
from db_building.AbfData import AbfData
from collections import Counter
from sklearn.metrics import confusion_matrix, balanced_accuracy_score

# Map target coa to correct index
TARGET_TO_INDEX = {'cOA4': 0,
                   'cOA5': 1,
                   'cOA6': 2}

INDEX_TO_TARGET = {v: k for k, v in TARGET_TO_INDEX.items()}

class CoaInference:
    """Class that uses generated CNNs to detect cOAs in abf files

    :param nn_dir: Path to directory that contains folders with cOA?/nn.h5 files
    :type nn_dir: Path
    """
    def __init__(self, nn_dir):
        self._nn_dir = nn_dir
        self.model_dict, self.input_width = self._load_models()

    def _load_models(self):
        """Called during __init__ and loads all CNNs into a model dictionary"""
        model_dict = {}
        all_model_files = list(self._nn_dir.glob('cOA?/nn.h5'))
        assert all_model_files, 'No generated neural networks were found'
        for f in all_model_files:
            # Extract target COA from folder name
            target = f.parts[-2]
            model = tf.keras.models.load_model(f, compile=False)
            model_dict[target] = model
        input_width = model.input_shape[1]
        return model_dict, input_width

    def predict_from_file(self, abf_path):
        """Counts cOAs in single input file

        :param abf_path: path to .abf file
        :type abf_path: str
        :return: Counter object that contains
        :rtype: Counter
        """
        abf = AbfData(abf_path)
        events = abf.get_pos(self.input_width)
        x_pad = np.expand_dims(events, -1)

        # ndarray, first axis is event no, second axis per-coa confidence score
        prediction = np.zeros((len(events), len(self.model_dict)))
        for target, model in self.model_dict.items():
            index = TARGET_TO_INDEX[target]
            prediction[:, index] = model.predict(x_pad).flatten()

        # Find which of the three models predicted the highest score
        highest_index = np.argmax(prediction, axis=1)
        labels = [INDEX_TO_TARGET[i] for i in highest_index]
        return Counter(labels)

def main(args):
    inference_model = CoaInference(args.nn_dir)

    y_true = []
    y_pred_list = []

    for abf in Path(args.abf_in).iterdir():
        # # Uncomment to omit the coa5 file
        # if 'cOA5' in abf.name:
        #     continue
        print(f'Processing {abf.name}')
        y_pred = inference_model.predict_from_file(str(abf))
        print(y_pred)
        true_coa = abf.name[:4]
        y_true.extend([true_coa] * sum(y_pred.values()))
        for coa_pred, count in y_pred.items():
            y_pred_list.extend([coa_pred] * count)

    print(confusion_matrix(y_true, y_pred_list))
    print('Balanced accuracy', balanced_accuracy_score(y_true, y_pred_list))

