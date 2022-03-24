import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import models, layers
from tensorflow.keras.models import load_model

from tensorflow.keras.metrics import CategoricalAccuracy, Precision, Recall
import numpy as np


class NeuralNetwork(object):
    """
    Convolutional Neural network to predict target coa presence in squiggle

    :param kernel_size: Kernel size of CNN
    :type kernel_size: int
    :param max_sequence_length: Maximum length of read that can be used to
    infer from. If read is longer than this length, don't use it.
    :type max_sequence_length: int
    :param weights: Path to h5 file that contains model weights, optional
    :param batch_size: Batch size to use during training
    :param threshold: Assign label to TRUE if probability above this threshold
                      when doing prediction
    :param eps_per_kmer_switch: Number of epochs to run
    :param filters: Number of CNN filters to use in the model
    :type filters: int
    :param learning_rate: Learning rate to use
    :param pool_size: If 0, do no pooling, else use this for the 1d maxpool
    """

    def __init__(self, **kwargs):
        self.filter_width = kwargs['filter_width']
        self.hfw = (self.filter_width - 1) // 2  # half filter width
        self.kernel_size = kwargs['kernel_size']
        self.max_sequence_length = kwargs['max_sequence_length']
        self.batch_size = kwargs['batch_size']
        self.threshold = kwargs['threshold']
        self.eps_per_kmer_switch = kwargs['eps_per_kmer_switch']
        self.filters = kwargs['filters']
        self.learning_rate = kwargs['learning_rate']
        self.pool_size = kwargs['pool_size']
        self.dropout_remove_prob = 1 - kwargs['dropout_keep_prob']
        self.num_layers = kwargs['num_layers']
        self.batch_norm = kwargs['batch_norm']

        self.initialize(kwargs['weights'])
        self.history = {'loss': [], 'binary_accuracy': [], 'precision': [],
                        'recall': [], 'val_loss': [], 'val_binary_accuracy': [],
                        'val_precision': [], 'val_recall': []}

    def initialize(self, weights=None):
        """Initialize the network.

        :param weights: Path to .h5 model summary with weights, optional.
                        If provided, use this to set the model weights
        """
        if weights:
            self.model = tf.keras.models.load_model(weights)

        # First layer
        self.model = models.Sequential()
        self.model.add(layers.Conv1D(self.filters,
                                     kernel_size=self.kernel_size,
                                     activation='relu',
                                     input_shape=(self.filter_width, 1),
                                     padding='valid'))
        for _ in range(self.num_layers):
            if self.batch_norm:
                self.model.add(layers.BatchNormalization())
            # self.model.add(layers.AvgPool1D(2))
            self.model.add(layers.Dropout(self.dropout_remove_prob))
            self.model.add(layers.Conv1D(self.filters,
                                         kernel_size=self.kernel_size,
                                         activation='relu'))
        if self.pool_size:
            self.model.add(layers.MaxPool1D(self.pool_size))
        # self.model.add(layers.GlobalMaxPool1D())
        self.model.add(layers.Flatten())
        self.model.add(layers.Dropout(self.dropout_remove_prob))
        self.model.add(layers.Dense(3, activation='softmax'))
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
                           loss=tf.keras.losses.CategoricalCrossentropy(),
                           metrics=[CategoricalAccuracy(), Precision(), Recall()])
        # if weights:
        #     self.model.load_weights(weights)
        #     print('Successfully loaded weights')

        # Uncomment to print model summary
        self.model.summary()

    def train(self, x, y, x_val, y_val, quiet=False, eps_per_kmer_switch=100):
        """Train the network. x_val/y_val may be used for validation/early
        stopping mechanisms.

        :param x: Input reads
        :param y: Ground truth labels of the read
        :param x_val: Input reads to use for validation
        :param y_val: Ground truth reads to use for validation
        :param quiet: If set to true, does not print to console
        """
        # Pad input sequences
        x_pad = np.expand_dims(pad_sequences(x, maxlen=self.filter_width,
                                             padding='post', truncating='post',
                                             dtype='float32'), -1)
        x_val_pad = np.expand_dims(pad_sequences(x_val,
                                                 maxlen=self.filter_width,
                                                 padding='post',
                                                 truncating='post',
                                                 dtype='float32'), -1)

        y = np.vstack(y)
        y_val = np.vstack(y_val)

        # Create tensorflow dataset
        tfd = tf.data.Dataset.from_tensor_slices((x_pad, y)).batch(
              self.batch_size).shuffle(x_pad.shape[0],
                                       reshuffle_each_iteration=True)

        # Train the model
        self.model.fit(tfd, epochs=self.eps_per_kmer_switch,
                       validation_data=(x_val_pad, y_val),
                       verbose=[2, 0][quiet])

    def predict(self, x, return_probs=False):
        """Given sequences input as x, predict if they contain target k-mer.
        Assumes the sequence x is a read that has been normalised,
        but not cut into smaller chunks.

        Function is mainly written to be called from train_nn.py.
        Not for final inference.

        :param x: Squiggle as numeric representation
        :type x: np.ndarray
        :param return_probs:
        :return: unnormalized predicted values
        :rtype: np.array of posteriors
        """

        x_pad = np.expand_dims(pad_sequences(x, maxlen=self.filter_width,
                                             padding='post',
                                             truncating='post',
                                             dtype='float32'), -1)

        posteriors = self.model.predict(x_pad)
        return np.argmax(posteriors, axis=-1)
