import argparse
import os

from functools import partial

import tensorflow as tf
import tensorflow.contrib.rnn as rnn

from data_input import read_dataset
from metrics import symmetric_mean_absolute_percent_error

parser = argparse.ArgumentParser()
parser.add_argument('--output_dir', type=str, help='output directory')


def model_fn(features, labels, mode, hyper_parameters):
    x = tf.split(features[hyper_parameters['time_series_name']], hyper_parameters['n_inputs'], 1)
    lstm_cell = rnn.BasicLSTMCell(hyper_parameters['n_units'], forget_bias=1.0)
    outputs, _ = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    outputs = outputs[-1]
    weight = tf.Variable(tf.random_normal([hyper_parameters['n_units'], hyper_parameters['n_outputs']]))
    bias = tf.Variable(tf.random_normal([hyper_parameters['n_outputs']]))
    predictions = tf.matmul(outputs, weight) + bias

    if mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL:
        loss = tf.losses.mean_squared_error(labels, predictions)
        train_op = tf.contrib.layers.optimize_loss(
            loss=loss,
            global_step=tf.train.get_global_step(),
            learning_rate=hyper_parameters['learning_rate'],
            optimizer="Adam")
        eval_metric_ops = {
            "rmse": tf.metrics.root_mean_squared_error(labels, predictions),
            "mae": tf.metrics.mean_absolute_error(labels, predictions),
            "smape": symmetric_mean_absolute_percent_error(labels, predictions)
        }
    else:
        loss = None
        train_op = None
        eval_metric_ops = None

    export_outputs = {'predict_output': tf.estimator.export.PredictOutput({"pred_output": predictions})}
    predictions_dict = {"predicted": predictions}

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions_dict,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=eval_metric_ops,
        export_outputs=export_outputs)


def train_input(training_dir, hyper_parameters):
    return read_dataset(os.path.join(training_dir, 'train-international-airline-passengers.csv'), hyper_parameters)


def eval_input(training_dir, hyper_parameters):
    return read_dataset(os.path.join(training_dir, 'valid-international-airline-passengers.csv'), hyper_parameters)


def serving_input_fn(hyper_parameters):
    feature_placeholders = {
        hyper_parameters['time_series_name']: tf.placeholder(tf.float32, [None, hyper_parameters['n_inputs']])
    }

    features = {
        key: tf.expand_dims(tensor, -1)
        for key, tensor in feature_placeholders.items()
    }

    features[hyper_parameters['time_series_name']] = tf.squeeze(features[hyper_parameters['time_series_name']],
                                                                axis=[2], name='timeseries')

    return tf.estimator.export.ServingInputReceiver(features, feature_placeholders)


def experiment_fn(output_dir, hyper_parameters):
    estimator = tf.estimator.Estimator(model_fn=partial(model_fn, hyper_parameters=hyper_parameters),
                                       model_dir=output_dir)

    early_stopping = tf.contrib.estimator.stop_if_no_decrease_hook(
        estimator,
        metric_name='loss',
        run_every_secs=60,
        max_steps_without_decrease=1000,
        min_steps=100)
    train_spec = tf.estimator.TrainSpec(input_fn=lambda: train_input('../data', hyper_parameters), max_steps=1000,
                                        hooks=[early_stopping])

    exporter = tf.estimator.FinalExporter('vanilla_lstm_predictor',
                                          lambda: serving_input_fn(hyper_parameters))
    eval_spec = tf.estimator.EvalSpec(input_fn=lambda: eval_input('../data', hyper_parameters),
                                      exporters=[exporter],
                                      steps=1)

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


def main(argv):
    tf.logging.set_verbosity(tf.logging.INFO)
    args = parser.parse_args(argv[1:])
    # shutil.rmtree(OUTPUT_DIR, ignore_errors=True) # start fresh each time

    hyper_parameters = {'learning_rate': 0.01, 'batch_size': 20, 'n_inputs': 3, 'n_outputs': 1, 'n_units': 3,
                        'time_series_name': 'passengers'}
    experiment_fn(args.output_dir, hyper_parameters)


if __name__ == "__main__":
    tf.app.run(main=main)
