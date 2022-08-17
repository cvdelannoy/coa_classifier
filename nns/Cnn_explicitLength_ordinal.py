import numpy as np
import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.metrics import CategoricalAccuracy, Precision, Recall
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nns.keras_metrics_from_logits import ordinal_accuracy
from sklearn.metrics import balanced_accuracy_score

from tensorflow.keras import backend
@tf.function
def cl2_accuracy(y_true, y_pred):
    cl = 1
    yh = backend.sum(backend.cast_to_floatx(backend.greater_equal(y_pred, 0.5)), 1)
    yt = backend.sum(backend.cast_to_floatx(backend.equal(y_true, 1)), 1)
    cl2_bool = backend.cast_to_floatx(backend.equal(yt, cl))
    result = tf.math.reduce_sum(tf.math.multiply(backend.cast_to_floatx(backend.equal(yt, yh)), cl2_bool)) / tf.math.reduce_sum(cl2_bool)
    return result

class NeuralNetwork(object):
    """
    Convolutional Neural network to predict target coa presence in squiggle

    :param kernel_size: Kernel size of CNN
    :type kernel_size: int
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
        # Filter_width can be set to false, which creates variable input length
        self.nb_classes = kwargs['nb_classes']
        self.label2class_dict = {}
        for ci in range(self.nb_classes):
            cur_lab = np.zeros(self.nb_classes-1, dtype=int)
            cur_lab[:ci] = 1
            self.label2class_dict[ci] = cur_lab
        self.filter_width = kwargs['filter_width']
        self.kernel_size = kwargs['kernel_size']
        self.batch_size = kwargs['batch_size']
        self.filters = kwargs['filters']
        self.learning_rate = kwargs['learning_rate']
        self.pool_size = kwargs['pool_size']
        self.dropout_remove_prob = 1 - kwargs['dropout_keep_prob']
        self.num_layers = kwargs['num_layers']
        self.batch_norm = kwargs['batch_norm']
        self.initialize(kwargs.get('weights'))
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
        input = tf.keras.Input((self.filter_width, 1))
        input_seqlen = tf.keras.Input((1))
        x = input
        # x_seqlen = tf.expand_dims(tf.tile(input_seqlen,(1,self.filter_width)), -1)
        # x = layers.concatenate((x, x_seqlen), axis=-1)
        x = layers.Conv1D(self.filters, kernel_size=self.kernel_size, activation='relu', padding='valid',
                          # kernel_initializer=tf.keras.initializers.zeros(), bias_initializer=tf.keras.initializers.zeros()
                          )(x)
        x_seqlen = tf.expand_dims(tf.tile(input_seqlen,(1,self.filter_width - self.kernel_size + 1)),-1)
        x = layers.concatenate((x, x_seqlen),axis=-1)
        for _ in range(self.num_layers):
            if self.batch_norm:
                x = layers.BatchNormalization()(x)
            x = layers.Dropout(self.dropout_remove_prob)(x)
            x = layers.Conv1D(self.filters, kernel_size=self.kernel_size, activation='relu',
                              # kernel_initializer=tf.keras.initializers.zeros(), bias_initializer=tf.keras.initializers.zeros()
                              )(x)
        if self.pool_size:
            x = layers.MaxPool1D(self.pool_size)(x)
        x = layers.Flatten()(x)
        x = layers.Dropout(self.dropout_remove_prob)(x)
        x = layers.concatenate((x, input_seqlen))
        x = layers.Dense(x.shape[1] + 1, activation='relu')(x)
        x = layers.Dense(self.nb_classes - 1, activation='sigmoid')(x)
        self.model = tf.keras.Model(inputs=[input, input_seqlen], outputs=x)

        def custom_loss_fun(y_actual, y_pred):
            hl = tf.keras.losses.Hinge()
            le = tf.keras.losses.MeanSquaredLogarithmicError()
            return le(y_actual, y_pred) + hl(y_actual, y_pred)

        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
                           # loss=tf.keras.losses.Hinge(),
                           # loss=tf.keras.losses.MeanSquaredLogarithmicError(),
                           loss=custom_loss_fun,
                           metrics=[ordinal_accuracy, cl2_accuracy])
        # if weights:
        #     self.model.load_weights(weights)
        #     print('Successfully loaded weights')

        # Uncomment to print model summary
        self.model.summary()

    def get_ordinal_labels(self, y):
        y_out = np.zeros((len(y), self.nb_classes-1), dtype=int)
        y_stacked = np.vstack(y)
        y_idx = np.argmax(y_stacked, axis=1)
        for yii, yi in enumerate(y_idx):
            y_out[yii,:yi] = 1
        return y_out

    def train(self, x, y, x_val, y_val, quiet=False, epochs=100):
        """Train the network. x_val/y_val may be used for validation/early
        stopping mechanisms.

        :param x: Input reads
        :param y: Ground truth labels of the read
        :param x_val: Input reads to use for validation
        :param y_val: Ground truth reads to use for validation
        :param quiet: If set to true, does not print to console
        """
        # Pad input sequences

        maxlen = self.filter_width if self.filter_width else None
        seq_lengths = np.expand_dims([len(xi) for xi in x], -1)
        seq_lengths_val = np.expand_dims([len(xi) for xi in x_val], -1)
        x_pad = np.expand_dims(pad_sequences(x, maxlen=maxlen,
                                             padding='post', truncating='post',
                                             dtype='float32'), -1)
        x_val_pad = np.expand_dims(pad_sequences(x_val,
                                                 maxlen=maxlen,
                                                 padding='post',
                                                 truncating='post',
                                                 dtype='float32'), -1)

        y = self.get_ordinal_labels(y)
        y_val = self.get_ordinal_labels(y_val)

        yt = y.copy()
        np.random.shuffle(yt)
        tst = cl2_accuracy(yt, y.astype('float32'))

        # Early stopping mechanism
        # callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
        #                                             patience=20)

        # Train the model
        self.model.fit(x={'input_1': x_pad, 'input_2': seq_lengths}, y=y, epochs=epochs,
                       validation_data=((x_val_pad, seq_lengths_val), y_val), shuffle=True,
                       verbose=[2, 0][quiet],
                       # callbacks=[callback]
                       )

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
        seq_lengths = np.expand_dims([len(xi) for xi in x], -1)

        posteriors = self.model.predict({'input_1': x_pad, 'input_2': seq_lengths})
        if return_probs:
            return posteriors

        yh = np.zeros(len(x), dtype=int) + self.nb_classes - 1
        marked_idx = []
        for ci in range(self.nb_classes-1):
            cur_marked = posteriors[:, ci] < 0.5
            for cmi, cm in enumerate(cur_marked):
                if cmi in marked_idx or not cm: continue
                yh[cmi] = ci
                marked_idx.append(cmi)

        return yh
