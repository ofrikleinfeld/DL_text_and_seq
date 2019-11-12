import random
import numpy as np

import mlp1 as mlp1
import train_loglin
from utils import F2I, L2I, TRAIN, DEV


STUDENT = {'name': "Ofri Kleinfeld",
         'ID': '302893680'}


def randomly_initialize_params(params):
    new_params = []
    for parameter in params:
        new_params.append(np.random.randn(*parameter.shape))
    return new_params


def accuracy_on_dataset(dataset, params):
    good = bad = 0.0
    for label, features in dataset:
        # YOUR CODE HERE
        # Compute the accuracy (a scalar) of the current parameters
        # on the dataset.
        # accuracy is (correct_predictions / all_predictions)

        pred_label = mlp1.predict(features, params)
        if pred_label == label:
            good += 1
        else:
            bad += 1

    return good / (good + bad)


def train_classifier(train_data, dev_data, num_iterations, learning_rate, params):
    """
    Create and train a classifier, and return the parameters.

    train_data: a list of (label, feature) pairs.
    dev_data  : a list of (label, feature) pairs.
    num_iterations: the maximal number of training iterations.
    learning_rate: the learning rate to use.
    params: list of parameters (initial values)
    """
    for i in range(num_iterations):
        cum_loss = 0.0  # total loss in this iteration.
        random.shuffle(train_data)
        for label, features in train_data:
            x = features
            y = label
            loss, grads = mlp1.loss_and_gradients(x, y, params)
            cum_loss += loss
            # YOUR CODE HERE
            # update the parameters according to the gradients
            # and the learning rate.

            new_params = []
            for j in range(len(params)):
                current_param = params[j]
                param_grad = grads[j]
                updated_param = current_param - learning_rate * param_grad
                new_params.append(updated_param)

            params = new_params

        train_loss = cum_loss / len(train_data)
        train_accuracy = accuracy_on_dataset(train_data, params)
        dev_accuracy = accuracy_on_dataset(dev_data, params)
        print(i, train_loss, train_accuracy, dev_accuracy)

    return params


if __name__ == '__main__':
    # YOUR CODE HERE
    # write code to load the train and dev sets, set up whatever you need,
    # and call train_classifier.

    # ...

    # load training and dev data
    train_data = [(L2I[l], train_loglin.feats_to_vec(bigrams)) for l, bigrams in TRAIN]
    dev_data = [(L2I[l], train_loglin.feats_to_vec(bigrams)) for l, bigrams in DEV]

    # define training hyper parameters
    in_dim = len(F2I)
    out_dim = len(L2I)
    hidden_dim = 50
    num_iterations = 50
    learning_rate = 1e-2

    # initiate classifier parameters
    params = mlp1.create_classifier(in_dim, hidden_dim, out_dim)
    params = randomly_initialize_params(params)

    # train
    trained_params = train_classifier(train_data, dev_data, num_iterations, learning_rate, params)
