import numpy as np

STUDENT = {'name': 'Ofri Kleinfeld',
         'ID': '302893680'}


class NetworkModule(object):
    def __init__(self):
        self.layer_input = None
        self.layer_output = None
        self.layer_grad = None

    def __call__(self, *args):
        raise NotImplementedError("Sub class must implement forward pass")

    def backward(self, *args):
        raise NotImplementedError("Sub class must implement backward pass")


class AbstractNetworkModuleWithParams(NetworkModule):
    def __init__(self):
        super().__init__()
        self.w = None
        self.b = None
        self.w_grad = None
        self.b_grad = None

    def init_weights(self, *args):
        raise NotImplementedError("Sub class must implement an initialization method for weights")

    def xavier_initialization(self, in_dimension, out_dimension):
        self.w = np.random.normal(loc=0, scale=in_dimension, size=(out_dimension, in_dimension))
        self.b = np.random.normal(loc=0, scale=in_dimension, size=out_dimension)


class Linear(AbstractNetworkModuleWithParams):
    def __init__(self, in_dimension, out_dimension):
        super().__init__()
        self.in_dim = in_dimension
        self.out_dim = out_dimension
        self.init_weights(self.in_dim, self.out_dim)

    def init_weights(self, in_dimension, out_dimension):
        self.xavier_initialization(in_dimension, out_dimension)

    def __call__(self, x):
        self.layer_input = x
        self.layer_output = np.einsum("i,ji->j", x, self.w) + self.b
        return self.layer_output

    def backward(self, next_layer_grad):
        self.layer_grad = np.einsum("j,ji->i", next_layer_grad, self.w)
        self.w_grad = np.einsum("j,i->ji", next_layer_grad, self.layer_input)
        self.b_grad = next_layer_grad
        return self.layer_grad


class Tanh(NetworkModule):
    @staticmethod
    def tanh_func(x):
        return np.tanh(x)

    @staticmethod
    def tanh_derivative(x):
        return 1 - np.tanh(x) ** 2

    def __call__(self, z):
        self.layer_input = z
        self.layer_output = self.tanh_func(self.layer_input)
        return self.layer_output

    def backward(self, next_layer_gard):
        self.layer_grad = next_layer_gard * self.tanh_derivative(self.layer_input)
        return self.layer_grad


class CELoss(NetworkModule):
    @staticmethod
    def softmax_func(x):
        z = x - np.max(x)
        exp_z = np.exp(z)

        return exp_z / sum(exp_z)

    @staticmethod
    def cross_entropy_loss(probs, y):
        loss = np.sum(-np.log(probs) * y)
        return loss

    def __init__(self):
        super().__init__()
        self.label = None
        self.loss = None

    def __call__(self, logits, label):
        self.layer_input = logits
        self.label = label
        self.layer_output = self.softmax_func(self.layer_input)

        self.loss = self.cross_entropy_loss(self.layer_output, self.label)

        return self.loss

    def backward(self):
        self.layer_grad = self.layer_output - self.label
        return self.layer_grad


def classifier_output(x, params):
    # YOUR CODE HERE.

    for layer in params:
        x = layer(x)

    logits = x
    return logits


def predict(x, params):
    return np.argmax(classifier_output(x, params))


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

    logits = classifier_output(x, params)
    y_one_hot = np.zeros(logits.shape)
    y_one_hot[y] = 1

    ce_loss = CELoss()
    loss = ce_loss(logits, y_one_hot)

    # backprop
    pred_grad = ce_loss.backward()
    for layer in params:
        layer.backward(pred_grad)
        pred_grad = layer.layer_grad

    # extract gradients w.r.t to parameters
    grads = []
    for layer in params:
        if isinstance(layer, AbstractNetworkModuleWithParams):
            grads.append(layer.w_grad)
            grads.append(layer.b_grad)

    return loss, grads


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

    layers_dims = [(in_dim, out_dim) for in_dim, out_dim in zip(dims[:-1], dims[1:])]
    for i in range(len(layers_dims) - 1):
        in_dim, out_dim = layers_dims[i]
        current_layer = Linear(in_dim, out_dim)
        current_activation = Tanh()
        params.append(current_layer)
        params.append(current_activation)

    hid_dim, pred_dim = layers_dims[-1]
    pred_layer = Linear(hid_dim, pred_dim)
    params.append(pred_layer)

    return params

