import os

from sagemaker.tensorflow import TensorFlow
from sagemaker.tuner import HyperparameterTuner, IntegerParameter, CategoricalParameter, ContinuousParameter


from utils import clean_bucket_prefix

BUCKET = ''
role = ''
source_dir = os.path.join(os.getcwd(), 'vanilla_lstm')

# clean_bucket_prefix(BUCKET, 'vanilla_lstm/checkpoints')

hyper_parameters = {'learning_rate': 0.01, 'batch_size': 20, 'n_inputs': 3, 'n_outputs': 1, 'n_units': 3,
                    'time_series_name': 'passengers'}

tf_estimator = TensorFlow(entry_point='sagemaker_estimator_adapter.py',
                          source_dir=source_dir,
                          base_job_name='vanilla-lstm-estimator', role=role,
                          training_steps=1000,
                          evaluation_steps=1,
                          hyperparameters=hyper_parameters,
                          train_instance_count=1, train_instance_type='ml.m5.large',
                          framework_version='1.11.0', py_version='py2',
                          checkpoint_path=('%s/vanilla_lstm/checkpoints' % BUCKET))

hyperparameter_ranges = {'learning_rate': ContinuousParameter(0.008, 0.2),
                         'n_inputs': IntegerParameter(3, 10),
                         'n_units': IntegerParameter(3, 10)}

objective_metric_name = 'mae'
objective_type = 'Minimize'
metric_definitions = [{'Name': 'mae',
                       'Regex': 'mae = ([0-9\\.]+)'}]


tuner = HyperparameterTuner(tf_estimator,
                            objective_metric_name,
                            hyperparameter_ranges,
                            metric_definitions,
                            max_jobs=40,
                            max_parallel_jobs=3,
                            objective_type=objective_type,
                            base_tuning_job_name='vanilla-lstm')

tuner.fit('%s/vanilla_lstm' % BUCKET)
