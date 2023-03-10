import tensorflow as tf
from tensorflow.keras import backend


@tf.function
def precision(y_true, y_pred):
    yh = backend.cast(backend.greater_equal(backend.sigmoid(y_pred), 0.5),
                      "float32")
    # If no positives predicted, but positives are present; return precision=0.0
    # if tf.math.count_nonzero(yh) == 0 and tf.math.count_nonzero(y_true) > 0:
    #     return 0.0
    result = backend.sum(y_true * yh) / backend.sum(yh)
    return result

@tf.function
def recall(y_true, y_pred):
    yh = backend.cast(backend.greater_equal(backend.sigmoid(y_pred), 0.5),
                      "float32")
    result = backend.sum(y_true * yh) / backend.sum(y_true)
    # assert not tf.math.is_nan(result)
    return result

@tf.function
def binary_accuracy(y_true, y_pred):
    yh = backend.cast(backend.greater_equal(backend.sigmoid(y_pred), 0.5),
                      "float32")
    result = backend.mean(backend.equal(y_true, yh))
    # assert not tf.math.is_nan(result)
    return result

@tf.function
def ordinal_accuracy(y_true, y_pred):
    yh = backend.sum(backend.cast_to_floatx(backend.greater_equal(y_pred, 0.5)), 1)
    yt = backend.sum(backend.cast_to_floatx(backend.equal(y_true, 1)), 1)
    # yh = backend.concatenate([backend.max(tf.where(x)) for x in backend.greater_equal(y_pred, 0.5)])
    # yt = backend.concatenate([backend.max(tf.where(x)) for x in backend.equal(y_true, 1)])
    result = backend.mean(backend.equal(yt, yh))
    return result
