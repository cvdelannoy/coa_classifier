from pathlib import Path
import tensorflow as tf
import numpy as np
from db_building.AbfData import AbfData
from collections import Counter

TARGET_TO_INDEX = {'cOA4': 0,
                   'cOA5': 1,
                   'cOA6': 2}

class Inference:
    def __init__(self, nn_dir):
        """

        :param nn_dir: Path to directory that contains folders with cOA?/nn.h5 files
        :type nn_dir: Path
        """
        self._nn_dir = nn_dir
        self.model_dict, self.input_width = self.load_models()

    def load_models(self):
        model_dict = {}
        all_model_files = list(self._nn_dir.glob('cOA?/nn.h5'))
        assert all_model_files, 'No generated neural networks were found'
        for f in all_model_files:
            target = f.parts[-2]
            model = tf.keras.models.load_model(f, compile=False)
            model_dict[target] = model
        input_width = model.input_shape[1]
        return model_dict, input_width

    def predict_from_file(self, abf_path):
        abf = AbfData(abf_path)
        events = abf.get_pos(self.input_width)
        x_pad = np.expand_dims(events, -1)
        prediction = np.zeros((len(events), len(self.model_dict)))
        for target, model in self.model_dict.items():
            index = TARGET_TO_INDEX[target]
            prediction[:, index] = model.predict(x_pad).flatten()

        highest_index = np.argmax(prediction, axis=1)
        return Counter(highest_index)

def main(args):
    inference_model = Inference(args.model)

    for abf in Path(args.abf_in).iterdir():
        print(abf.name)
        print(inference_model.predict_from_file(abf))
        print('')
