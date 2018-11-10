import os

from sagemaker.tensorflow import TensorFlow

from utils import clean_bucket_prefix

BUCKET = ''
role = ''
source_dir = os.path.join(os.getcwd(), 'vanilla_lstm')

clean_bucket_prefix(BUCKET, 'vanilla_lstm/checkpoints')

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

tf_estimator.fit('%s/vanilla_lstm' % BUCKET)
