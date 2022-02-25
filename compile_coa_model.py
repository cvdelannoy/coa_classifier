import os, yaml, h5py

from pathlib import Path
import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))


def compile_model_abundance(mod_dict, filter_width, filter_stride, threshold):
    input = tf.keras.Input(shape=(None, 1), ragged=True)
    input_strided = tf.signal.frame(input.to_tensor(default_value=np.nan), frame_length=filter_width, frame_step=filter_stride, axis=1)
    input_strided = tf.keras.layers.Masking(mask_value=np.nan)(input_strided)
    ht_list = []
    for km in mod_dict:
        mod = tf.keras.models.load_model(f'{mod_dict[km]}/nn.h5',compile=False)
        mod._name = km
        h = tf.keras.layers.TimeDistributed(mod)(input_strided)
        h = K.cast_to_floatx(K.greater(h, threshold))
        h = K.sum(h, axis=0)
        h = K.sum(h, axis=0)
        ht_list.append(h)
    output = tf.keras.layers.concatenate(ht_list)
    meta_mod = tf.keras.Model(inputs=input, outputs=output)
    meta_mod.compile()
    return meta_mod


def main(args):

    # List for which compounds models are available
    mod_dict = {pth.name: str(pth) for pth in Path(args.nn_directory).iterdir() if pth.is_dir()}

    # Parse target k-mers
    with open(args.parameter_file, 'r') as fh:
        param_dict = yaml.load(fh, yaml.FullLoader)
    mod = compile_model_abundance(mod_dict,
                                  param_dict['filter_width'], param_dict['filter_stride'],
                                  param_dict['threshold'])
    mod.save(args.out_model)
    with h5py.File(args.out_model, 'r+') as fh:
        fh.attrs['compound_list'] = ','.join(list(mod_dict))
