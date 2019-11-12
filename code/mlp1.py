import numpy as np

from loglinear import softmax, cross_entropy_loss

STUDENT = {'name': 'Ofri Kleinfeld',
         'ID': '302893680'}


def tanh(x):
    return np.tanh(x)


def tanh_derivative(x):
    return 1 - tanh(x) ** 2


def mat_vec_mul(vec, mat):
    return np.einsum("i,ij->j", vec, mat)


def mat_vec_mul_reverse(vec, mat):
    return np.einsum("j,ij->i", vec, mat)


def vec_and_vec_to_mat_mul(matrix_columns, matrix_rows):
    return np.einsum("j,i->ij", matrix_columns, matrix_rows)


def classifier_output(x, params):
    # YOUR CODE HERE.
    W, b, U, b_tag = params

    z = mat_vec_mul(x, W) + b
    h = tanh(z)
    o = mat_vec_mul(h, U) + b_tag
    probs = softmax(o)

    return probs, [z, h, o]


def predict(x, params):
    """
    params: a list of the form [W, b, U, b_tag]
    """
    probs, _ = classifier_output(x, params)
    return np.argmax(probs)


def loss_and_gradients(x, y, params):
    """
    params: a list of the form [W, b, U, b_tag]

    returns:
        loss,[gW, gb, gU, gb_tag]

    loss: scalar
    gW: matrix, gradients of W
    gb: vector, gradients of b
    gU: matrix, gradients of U
    gb_tag: vector, gradients of b_tag
    """
    # YOU CODE HERE
    W, b, U, b_tag = params

    probs, hiddens = classifier_output(x, params)
    z, h, o = hiddens

    y_one_hot = np.zeros(probs.shape)
    y_one_hot[y] = 1
    loss = cross_entropy_loss(probs, y)

    # backprop part
    # gradients with respect to layers inputs
    g_o = probs - y_one_hot
    g_h = mat_vec_mul_reverse(g_o, U)
    g_z = g_h * tanh_derivative(h)

    # gradients with respect to parameters
    gW = vec_and_vec_to_mat_mul(g_z, x)
    gb = g_z
    gU = vec_and_vec_to_mat_mul(g_o, h)
    gb_tag = g_o

    return loss, [gW, gb, gU, gb_tag]


def create_classifier(in_dim, hid_dim, out_dim):
    """
    returns the parameters for a multi-layer perceptron,
    with input dimension in_dim, hidden dimension hid_dim,
    and output dimension out_dim.

    return:
    a flat list of 4 elements, W, b, U, b_tag.
    """
    W = np.zeros((in_dim, hid_dim))
    b = np.zeros(hid_dim)
    U = np.zeros((hid_dim, out_dim))
    b_tag = np.zeros(out_dim)

    params = [W, b, U, b_tag]
    return params

