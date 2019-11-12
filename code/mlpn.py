import numpy as np

from train_mlp1 import randomly_initialize_params
import mlp1
import loglinear

STUDENT = {'name': 'Ofri Kleinfeld',
         'ID': '302893680'}


def classifier_output(x, params):
    # YOUR CODE HERE.

    current_param = 0
    current_input = x
    hidden_layers = []
    while current_param < len(params):
        W = params[current_param]
        b = params[current_param + 1]
        z = mlp1.mat_vec_mul(current_input, W) + b

        hidden_layers.append(z)
        current_input = z
        # don't do non linearity in last hidden representation before softmax
        if current_param + 2 < len(params):
            h = mlp1.tanh(z)
            hidden_layers.append(h)
            current_input = h

        current_param += 2

    probs = loglinear.softmax(current_input)
    return probs, hidden_layers


def predict(x, params):
    probs, _ = classifier_output(x, params)
    return np.argmax(probs)


def loss_and_gradients(x, y, params):
    """
    params: a list as created by create_classifier(...)

    returns:
        loss,[gW1, gb1, gW2, gb2, ...]

    loss: scalar
    gW1: matrix, gradients of W1
    gb1: vector, gradients of b1
    gW2: matrix, gradients of W2
    gb2: vector, gradients of b2
    ...

    (of course, if we request a linear classifier (ie, params is of length 2),
    you should not have gW2 and gb2.)
    """
    # YOU CODE HERE

    probs, hidden_layers = classifier_output(x, params)
    y_one_hot = np.zeros(probs.shape)
    y_one_hot[y] = 1
    loss = loglinear.cross_entropy_loss(probs, y)

    # backprop part
    # gradients with respect to layers inputs

    layer_gradients = []
    last_hidden_grad = probs - y_one_hot
    layer_gradients.append(last_hidden_grad)
    
    g_h = mat_vec_mul_reverse(g_o, U)
    g_z = g_h * tanh_derivative(z)


    return ...


def create_classifier(dims):
    """
    returns the parameters for a multi-layer perceptron with an arbitrary number
    of hidden layers.
    dims is a list of length at least 2, where the first item is the input
    dimension, the last item is the output dimension, and the ones in between
    are the hidden layers.
    For example, for:
        dims = [300, 20, 30, 40, 5]
    We will have input of 300 dimension, a hidden layer of 20 dimension, passed
    to a layer of 30 dimensions, passed to learn of 40 dimensions, and finally
    an output of 5 dimensions.
    
    Assume a tanh activation function between all the layers.

    return:
    a flat list of parameters where the first two elements are the W and b from input
    to first layer, then the second two are the matrix and vector from first to
    second layer, and so on.
    """
    params = []

    params_shapes = [((in_dim, out_dim), out_dim) for in_dim, out_dim in zip(dims[:-1], dims[1:])]
    for w_shape, b_shape in params_shapes:
        params.append(np.zeros(*w_shape))
        params.append(np.zeros(*b_shape))

    params = randomly_initialize_params(params)
    return params

