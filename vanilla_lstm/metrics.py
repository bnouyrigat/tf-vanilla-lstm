import tensorflow as tf
from tensorflow.python.ops.confusion_matrix import remove_squeezable_dimensions


def mean_absolute_error(labels,
                        predictions,
                        weights=None,
                        metrics_collections=None,
                        updates_collections=None,
                        name=None):
    if tf.executing_eagerly():
        raise RuntimeError('mean_absolute_error is not supported when eager execution is enabled.')

    predictions, labels = remove_squeezable_dimensions(predictions=predictions, labels=labels)
    absolute_errors = tf.abs(predictions - labels)
    return tf.metrics.mean(absolute_errors, weights, metrics_collections, updates_collections,
                           name or 'mean_absolute_error')


def symmetric_mean_absolute_percent_error(labels, predictions):
    if tf.executing_eagerly():
        raise RuntimeError('symmetric_mean_absolute_percent_error is not supported '
                           'when eager execution is enabled.')

    predictions_, labels_ = remove_squeezable_dimensions(predictions=predictions, labels=labels)

    denominator = (tf.abs(predictions_) + tf.abs(labels_)) / 2.0
    diff = tf.div(tf.abs(predictions_ - labels_), denominator)
    mask = tf.where(denominator == 0, tf.ones_like(denominator), tf.zeros_like(denominator))
    # return tf.metrics.mean(tf.multiply(diff, mask))
    return tf.metrics.mean(diff, name='symmetric_mean_absolute_percent_error')
