import random
import numpy as np

import loglinear as ll
from utils import F2I, L2I, TRAIN, DEV


STUDENT = {'name': "Ofri Kleinfeld",
         'ID': '302893680'}


def feats_to_vec(features):
    # YOUR CODE HERE.
    # Should return a numpy vector of features.
    f_vec = np.zeros(len(F2I))
    for bigram in features:
        f_index = F2I.get(bigram, None)
        if f_index is not None:
            f_vec[f_index] += 1

    return f_vec


def accuracy_on_dataset(dataset, params):
    good = bad = 0.0
    for label, features in dataset:
        # YOUR CODE HERE
        # Compute the accuracy (a scalar) of the current parameters
        # on the dataset.
        # accuracy is (correct_predictions / all_predictions)

        pred_label = ll.predict(features, params)
        good += 1 if pred_label == label else bad +1

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
            x = feats_to_vec(features) # convert features to a vector.
            y = label                  # convert the label to number if needed.
            loss, grads = ll.loss_and_gradients(x, y, params)
            cum_loss += loss
            # YOUR CODE HERE
            # update the parameters according to the gradients
            # and the learning rate.

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
    train_data = [(L2I[l], feats_to_vec(bigrams)) for l, bigrams in TRAIN]
    dev_data = [(L2I[l], feats_to_vec(bigrams)) for l, bigrams in DEV]

    # define training hyper parameters
    in_dim = len(F2I)
    out_dim = len(L2I)
    num_iterations = 10
    learning_rate = 1e-3

    # initiate classifier parameters
    W, b = ll.create_classifier(in_dim, out_dim)
    W = np.random.randn(W.shape[0], W.shape[1])
    b = np.random.randn(b.shape[0])

    # train
    trained_params = train_classifier(train_data, dev_data, num_iterations, learning_rate, params)

