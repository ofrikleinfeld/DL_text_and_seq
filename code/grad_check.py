import numpy as np

STUDENT={'name': 'Ofri Kleinfeld',
         'ID': '302893680'}


def gradient_check(f, x):
    """ 
    Gradient check for a function f 
    - f should be a function that takes a single argument and outputs the cost and its gradients
    - x is the point (numpy array) to check the gradient at
    """ 
    fx, grad = f(x)  # Evaluate function value at original point
    h = 1e-4

    # Iterate over all indexes in x
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index

        ### modify x[ix] with h defined above to compute the numerical gradient.
        ### if you change x, make sure to return it back to its original state for the next iteration.
        ### YOUR CODE HERE:
        x_plus_h = np.copy(x)
        x_plus_h[ix] += h
        fx_plus_h, _ = f(x_plus_h)

        x_minus_h = np.copy(x)
        x_minus_h[ix] -= h
        fx_minus_h, _ = f(x_minus_h)

        numeric_gradient = (fx_plus_h - fx_minus_h) / (2 * h)
        ### END YOUR CODE

        # Compare gradients
        reldiff = abs(numeric_gradient - grad[ix]) / max(1, abs(numeric_gradient), abs(grad[ix]))
        if reldiff > 1e-5:
            print("Gradient check failed.")
            print("First gradient error found at index %s" % str(ix))
            print("Your gradient: %f \t Numerical gradient: %f" % (grad[ix], numeric_gradient))
            return
    
        it.iternext()  # Step to next index

    print("Gradient check passed!")


def sanity_check():
    """
    Some basic sanity checks.
    """
    quad = lambda x: (np.sum(x ** 2), x * 2)

    print("Running sanity checks...")
    gradient_check(quad, np.array(123.456))      # scalar test
    gradient_check(quad, np.random.randn(3,))    # 1-D test
    gradient_check(quad, np.random.randn(4, 5))   # 2-D test
    print("")


def tanh_derivative_check():
    from mlp1 import tanh, tanh_derivative

    tanh_f = lambda x: (np.sum(tanh(x)), tanh_derivative(x))

    print("Checking tanh function gradient")
    gradient_check(tanh_f, np.array(123.456))      # scalar test
    gradient_check(tanh_f, np.random.randn(3,))    # 1-D test
    gradient_check(tanh_f, np.random.randn(4, 5))   # 2-D test
    print("")


def mlp_check():
    from mlp1 import create_classifier, loss_and_gradients as mlp1_loss_and_grad
    from train_mlp1 import randomly_initialize_params
    in_dim, hid_dim, out_dim = 5, 3, 2
    initialized_params = create_classifier(in_dim=in_dim, hid_dim=hid_dim, out_dim=out_dim)

    x = np.random.randn(in_dim, )
    y = 0
    for i in range(5):
        random_params = randomly_initialize_params(initialized_params)
        W, b, U, b_tag = random_params

        def _loss_and_W_grad(W_):
            loss, grads = mlp1_loss_and_grad(x, y, [W_, b, U, b_tag])
            return loss, grads[0]

        def _loss_and_b_grad(b_):
            loss, grads = mlp1_loss_and_grad(x, y, [W, b_, U, b_tag])
            return loss, grads[1]

        def _loss_and_U_grad(U_):
            loss, grads = mlp1_loss_and_grad(x, y, [W, b, U_, b_tag])
            return loss, grads[2]

        def _loss_and_b_tag_grad(b_tag_):
            loss, grads = mlp1_loss_and_grad(x, y, [W, b, U, b_tag_])
            return loss, grads[3]

        print(f"Gradients checks for random initialization {i+1}")
        gradient_check(_loss_and_W_grad, W)
        gradient_check(_loss_and_b_grad, b)
        gradient_check(_loss_and_U_grad, U)
        gradient_check(_loss_and_b_tag_grad, b_tag)


def mlpn_loglinear_check():
    import mlpn
    import train_mlp1
    dims = [20, 3]
    params = mlpn.create_classifier(dims)

    x = np.random.randn(dims[0], )
    y = 0

    print("MLP arbitrary layers gradient check (special case of log linear model")
    for i in range(5):
        random_params = train_mlp1.randomly_initialize_params(params)
        W, b = random_params

        def _loss_and_W_grad(W_):
            loss, grads = mlpn.loss_and_gradients(x, y, [W_, b])
            return loss, grads[0]

        def _loss_and_b_grad(b_):
            loss, grads = mlpn.loss_and_gradients(x, y, [W, b_])
            return loss, grads[1]

        print(f"Gradients checks for random initialization {i + 1}")
        gradient_check(_loss_and_W_grad, W)
        gradient_check(_loss_and_b_grad, b)


def mlpn_2_hidden_check():
    import mlpn
    import train_mlp1
    dims = [50, 20, 10, 3]
    params = mlpn.create_classifier(dims)

    x = np.random.randn(dims[0], )
    y = 0

    print("MLP arbitrary layers gradient check (special case of log linear model")
    for i in range(5):
        random_params = train_mlp1.randomly_initialize_params(params)
        W1, b1, W2, b2, W3, b3 = random_params

        def _loss_and_W1_grad(W1_):
            loss, grads = mlpn.loss_and_gradients(x, y, [W1_, b1, W2, b2, W3, b3])
            return loss, grads[0]

        def _loss_and_b1_grad(b1_):
            loss, grads = mlpn.loss_and_gradients(x, y, [W1, b1_, W2, b2, W3, b3])
            return loss, grads[1]

        def _loss_and_W2_grad(W2_):
            loss, grads = mlpn.loss_and_gradients(x, y, [W1, b1, W2_, b2, W3, b3])
            return loss, grads[2]

        def _loss_and_b2_grad(b2_):
            loss, grads = mlpn.loss_and_gradients(x, y, [W1, b1, W2, b2_, W3, b3])
            return loss, grads[3]

        def _loss_and_W3_grad(W3_):
            loss, grads = mlpn.loss_and_gradients(x, y, [W1, b1, W2, b2, W3_, b3])
            return loss, grads[4]

        def _loss_and_b3_grad(b3_):
            loss, grads = mlpn.loss_and_gradients(x, y, [W1, b1, W2, b2, W3, b3_])
            return loss, grads[5]

        print(f"Gradients checks for random initialization {i + 1}")
        gradient_check(_loss_and_W1_grad, W1)
        gradient_check(_loss_and_b1_grad, b1)
        gradient_check(_loss_and_W2_grad, W2)
        gradient_check(_loss_and_b2_grad, b2)
        gradient_check(_loss_and_W3_grad, W3)
        gradient_check(_loss_and_b3_grad, b3)


def mlpn_3_hidden_check():
    import mlpn
    import train_mlp1
    dims = [50, 30, 20, 10, 3]
    params = mlpn.create_classifier(dims)

    x = np.random.randn(dims[0], )
    y = 0

    print("MLP arbitrary layers gradient check (special case of log linear model")
    for i in range(5):
        random_params = train_mlp1.randomly_initialize_params(params)
        W1, b1, W2, b2, W3, b3, W4, b4 = random_params

        def _loss_and_W1_grad(W1_):
            loss, grads = mlpn.loss_and_gradients(x, y, [W1_, b1, W2, b2, W3, b3, W4, b4])
            return loss, grads[0]

        def _loss_and_b1_grad(b1_):
            loss, grads = mlpn.loss_and_gradients(x, y, [W1, b1_, W2, b2, W3, b3, W4, b4])
            return loss, grads[1]

        def _loss_and_W2_grad(W2_):
            loss, grads = mlpn.loss_and_gradients(x, y, [W1, b1, W2_, b2, W3, b3, W4, b4])
            return loss, grads[2]

        def _loss_and_b2_grad(b2_):
            loss, grads = mlpn.loss_and_gradients(x, y, [W1, b1, W2, b2_, W3, b3, W4, b4])
            return loss, grads[3]

        def _loss_and_W3_grad(W3_):
            loss, grads = mlpn.loss_and_gradients(x, y, [W1, b1, W2, b2, W3_, b3, W4, b4])
            return loss, grads[4]

        def _loss_and_b3_grad(b3_):
            loss, grads = mlpn.loss_and_gradients(x, y, [W1, b1, W2, b2, W3, b3_, W4, b4])
            return loss, grads[5]

        def _loss_and_W4_grad(W4_):
            loss, grads = mlpn.loss_and_gradients(x, y, [W1, b1, W2, b2, W3, b3, W4_, b4])
            return loss, grads[6]

        def _loss_and_b4_grad(b4_):
            loss, grads = mlpn.loss_and_gradients(x, y, [W1, b1, W2, b2, W3, b3, W4, b4_])
            return loss, grads[7]

        print(f"Gradients checks for random initialization {i + 1}")
        gradient_check(_loss_and_W1_grad, W1)
        gradient_check(_loss_and_b1_grad, b1)
        gradient_check(_loss_and_W2_grad, W2)
        gradient_check(_loss_and_b2_grad, b2)
        gradient_check(_loss_and_W3_grad, W3)
        gradient_check(_loss_and_b3_grad, b3)
        gradient_check(_loss_and_W4_grad, W4)
        gradient_check(_loss_and_b4_grad, b4)


if __name__ == '__main__':
    # If these fail, your code is definitely wrong.
    sanity_check()
    tanh_derivative_check()
    mlp_check()
    mlpn_loglinear_check()
    mlpn_2_hidden_check()
    mlpn_3_hidden_check()
