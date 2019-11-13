import numpy as np

from train_mlp1 import randomly_initialize_params
import mlp1
import loglinear

STUDENT = {'name': 'Ofri Kleinfeld',
         'ID': '302893680'}


def classifier_output(x, params):
    # YOUR CODE HERE.

    current_input = x
    hidden_outputs = [current_input]

    pred_W, pred_b = params[-2:]
    hidden_params = params[:-2]
    for i in range(0, len(hidden_params), 2):
        W = hidden_params[i]
        b = hidden_params[i+1]
        z = mlp1.mat_vec_mul(current_input, W) + b
        h = mlp1.tanh(z)

        current_input = h
        hidden_outputs.append(z)
        hidden_outputs.append(h)

    logits = mlp1.mat_vec_mul(current_input, pred_W) + pred_b
    probs = loglinear.softmax(logits)
    return probs, hidden_outputs


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

    probs, hidden_outputs = classifier_output(x, params)
    y_one_hot = np.zeros(probs.shape)
    y_one_hot[y] = 1
    loss = loglinear.cross_entropy_loss(probs, y)

    # backprop part
    # gradients with respect to layers inputs
    logits_grad = probs - y_one_hot
    layers_grad = [logits_grad]

    # special case last layer no activation
    last_layer_grad = layers_grad[-1]
    last_W, _ = params[-2:]
    last_hidden_grad = mlp1.mat_vec_mul_reverse(last_layer_grad, last_W)
    layers_grad.append(last_hidden_grad)

    for i in range(len(params[:-2]), 0, -2):
        current_W = params[i-2]
        last_layer_grad = layers_grad[-1]
        z_layer_grad = last_layer_grad * mlp1.tanh_derivative(hidden_outputs[i-1])
        h_layer_grad = mlp1.mat_vec_mul_reverse(z_layer_grad, current_W)

        layers_grad.append(z_layer_grad)
        layers_grad.append(h_layer_grad)

    # gradients with respect to parameters
    param_grads = []
    reversed_layer_grad = layers_grad[::-1]
    linear_layer_grads = reversed_layer_grad[1::2]
    linear_hidden_outputs = [hidden_outputs[0]] + hidden_outputs[1::2]
    for i in range(len(linear_hidden_outputs)):
        current_grad = linear_layer_grads[i]
        current_output = linear_hidden_outputs[i]
        gw_i = mlp1.vec_and_vec_to_mat_mul(current_grad, current_output)
        gb_i = current_grad

        param_grads.append(gw_i)
        param_grads.append(gb_i)

    return loss, param_grads


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

    params_shapes = [(in_dim, out_dim) for in_dim, out_dim in zip(dims[:-1], dims[1:])]
    for in_dim, out_dim in params_shapes:
        params.append(np.zeros((in_dim, out_dim)))
        params.append(np.zeros(out_dim))

    params = randomly_initialize_params(params)
    return params

