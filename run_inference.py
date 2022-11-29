import os, yaml, h5py, ast
from collections import Counter
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, balanced_accuracy_score
from tensorflow.keras.preprocessing.sequence import pad_sequences

import matplotlib.pyplot as plt

from db_building.AbfData import AbfData
from resources.helper_functions import parse_output_path, parse_input_path
# tf.config.set_visible_devices(gpus[1:], 'GPU')


class CoaInference:
    """Class that uses generated CNNs to detect cOAs in abf files

    :param nn_dir: Path to directory that contains folders with nn.h5 file
    :type nn_dir: Path
    """
    def __init__(self, nn):
        self.nn = tf.keras.models.load_model(nn)
        with h5py.File(nn, 'r') as fh:
            self.event_type_dict = ast.literal_eval(fh.attrs['event_type_dict'])
        input_shapes = self.nn.input_shape
        if type(input_shapes) == list:  # input consists of multiple tensors  -> take first
            self.input_length = input_shapes[0][1]
        elif type(input_shapes) == tuple:  # input consists of single tensor
            self.input_length = input_shapes[1]
        self.target_to_index = {x: i for i, x in enumerate(np.unique(list(self.event_type_dict.values())))}
        self.index_to_target = {v: k for k, v in self.target_to_index.items()}

    def predict_from_file(self, abf_path, bootstrap=False, return_traces=False):
        """Counts cOAs in single input file

        :param abf_path: path to .abf file
        :type abf_path: str
        :param bootstrap: indicate if .abf file should be bootstrapped
        :type bootstrap: bool
        :return: Counter object that contains
        :rtype: Counter
        """
        abf = AbfData(abf_path, normalization='median',
                      lowpass_freq=80,
                      baseline_fraction=0.65,
                      event_type_dict=self.event_type_dict)
        events = abf.get_pos(unfiltered=False)
        x_pad = np.expand_dims(pad_sequences(events, maxlen=self.input_length,
                                             padding='post', truncating='post',
                                             dtype='float32'), -1)
        if bootstrap:
            x_pad = x_pad[np.random.randint(0, len(x_pad), size=len(x_pad))]
        if len(self.nn.inputs) == 2:
            x_lens = np.array([len(x) for x in events])
            y_hat = self.nn.predict([x_pad, x_lens])
        else:
            y_hat = self.nn.predict(x_pad)
        y_hat = np.argmax(y_hat, axis=1)
        y_pred = [self.index_to_target[i] for i in y_hat]
        if return_traces:
            return y_pred, x_pad
        return y_pred


def main(args):
    # if args.no_gpu:
    #     # Limit resources
    #     os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    #     os.environ["OMP_NUM_THREADS"] = "1"
    #     os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
    #     os.environ["TF_NUM_INTEROP_THREADS"] = "1"
    #     tf.config.threading.set_intra_op_parallelism_threads(1)
    #     tf.config.threading.set_inter_op_parallelism_threads(1)
    #     tf.config.set_soft_device_placement(True)

    inference_model = CoaInference(args.nn_path)

    y_true = []
    y_pred_list = []

    abf_list = parse_input_path(args.abf_in, pattern='*.abf') + parse_input_path(args.abf_in, pattern='*.npz')

    for i, abf in enumerate(abf_list):
        abf = Path(abf)
        print(f'Processing {abf.name}')
        y_pred = inference_model.predict_from_file(abf, args.bootstrap, args.save_traces)
        if args.save_traces:
            y_pred, x_traces = y_pred
        y_pred_list.extend(y_pred)
        label_list = list(inference_model.index_to_target.values())
        true_coa = inference_model.event_type_dict.get(abf.name[:4].lower(), abf.name[:4].lower())
        if not true_coa.lower().startswith('coa'):
            # Looking at file of unknown type
            true_coa = 'UNKNOWN'
            label_list += ['UNKNOWN']
        if args.save_traces:
            abf_id = os.path.splitext(abf.name)[0]
            plot_dir_dict = {class_id: parse_output_path(f'{args.out_dir}{abf_id}/{class_id}/')
                             for class_id in np.unique(y_pred)}
            fig, (ax_t, ax_f) = plt.subplots(2,1, figsize=(10,10))
            x = np.arange(x_traces.shape[1])
            for ti, (xt, yp) in enumerate(zip(x_traces, y_pred)):
                fig_single, ax = plt.subplots(figsize=(10, 5))
                ax.plot(x, xt)
                fig_single.savefig(f'{plot_dir_dict[yp]}{ti}.svg')
                plt.close(fig_single)
                if yp == true_coa:
                    ax_t.plot(x,xt)
                else:
                    ax_f.plot(x, xt)
            fig.savefig(f'{args.out_dir}{abf_id}/all_traces.svg')
            plt.close(fig)
            y_pred_idx = [f'{yi},{yp}' for yi, yp in enumerate(y_pred)]
            with open(f'{args.out_dir}{abf_id}/prediction.csv', 'w') as fh:
                fh.write('\n'.join(y_pred_idx))
            np.save(f'{args.out_dir}{abf_id}/traces.npy', x_traces)
        y_true.extend([true_coa] * len(y_pred))
        # print("Y pred", y_pred)
        # print('Y true', y_true)
    conf_mat = confusion_matrix(y_true, y_pred_list, labels=label_list)
    np.savetxt(args.out_dir + 'confmat.csv', conf_mat)
    with open(args.out_dir + 'confmat_labels.txt', 'w') as fh: fh.write('\n'.join(list(label_list)))
    pred_counts = Counter(y_pred_list)

    with open(args.out_dir + 'summary_stats.yaml', 'w') as fh:
        fh.write(f'balanced_accuracy: {balanced_accuracy_score(y_true, y_pred_list)}\n')
        for key, value in pred_counts.items():
            fh.write(f"{key}: {value}\n")
    print(conf_mat)
    print('Balanced accuracy', balanced_accuracy_score(y_true, y_pred_list))

