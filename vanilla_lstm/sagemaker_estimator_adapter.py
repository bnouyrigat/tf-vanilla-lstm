import vanilla_lstm_estimator as estimator


def model_fn(features, labels, mode, hyperparameters):
    return estimator.model_fn(features, labels, mode, hyperparameters)


def train_input_fn(training_dir, hyperparameters):
    return estimator.train_input(training_dir, hyperparameters)


def eval_input_fn(training_dir, hyperparameters):
    return estimator.eval_input(training_dir, hyperparameters)


def serving_input_fn(hyperparameters):
    return estimator.serving_input_fn(hyperparameters)
