import numpy as np
import pandas as pd
import tensorflow as tf


def easy_input_function(dataframe, series_key, batch_size, window_size):
    series = dataframe[series_key]
    x, y = build_sequences(series, windows=window_size, stride=1)
    ds = tf.data.Dataset.from_tensor_slices(({series_key: x}, y))
    ds = ds.batch(batch_size).repeat(count=None)
    return ds


def build_sequences(dataset, windows=5, stride=3):
    dataset_x, dataset_y = [], []
    n_train = dataset.shape[0]
    sequence_length = (n_train - windows - stride + 1)

    for i in range(sequence_length):
        dataset_x.append(dataset[i:(i + windows)])
        dataset_y.append(dataset[i + windows: i + windows + stride])
    return np.array(dataset_x, dtype=np.float32), np.array(dataset_y, dtype=np.float32)


def read_dataset(filename, hyper_parameters):
    df = pd.read_csv(filepath_or_buffer=filename,
                     header=0,
                     index_col='Month',
                     parse_dates=True,
                     infer_datetime_format=True)
    return easy_input_function(dataframe=df,
                               series_key=hyper_parameters['time_series_name'],
                               batch_size=hyper_parameters['batch_size'],
                               window_size=hyper_parameters['n_inputs'])
