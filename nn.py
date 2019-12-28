import numpy as np

from submission import Submission
from logreg import sigmoid
from utils import add_intercept


class NNGrader(Submission):
    X = np.reshape(3 * np.sin(np.arange(1, 31)), (3, 10), order='F')
    Xm = np.reshape(np.sin(np.arange(1, 33)), (16, 2), order='F') / 5
    ym = np.arange(1, 17) % 4
    t1 = np.sin(np.reshape(np.arange(1, 25, 2), (4, 3), order='F'))
    t2 = np.cos(np.reshape(np.arange(1, 41, 2), (4, 5), order='F'))
    t = np.concatenate([t1.ravel(), t2.ravel()], axis=0)

    def __init__(self):
        part_names = ['Feedforward and Cost Function',
                      'Regularized Cost Function',
                      'Sigmoid Gradient',
                      'Neural Network Gradient (Backpropagation)',
                      'Regularized Gradient']
        super().__init__('neural-network-learning', part_names)

    def __iter__(self):
        for part_id in range(1, 6):
            try:
                func = self.functions[part_id]

                # Each part has different expected arguments/different function
                if part_id == 1:
                    res = func(self.t, 2, 4, 4, self.Xm, self.ym, 0)[0]
                elif part_id == 2:
                    res = func(self.t, 2, 4, 4, self.Xm, self.ym, 1.5)
                elif part_id == 3:
                    res = func(self.X, )
                elif part_id == 4:
                    J, grad = func(self.t, 2, 4, 4, self.Xm, self.ym, 0)
                    grad1 = np.reshape(grad[:12], (4, 3))
                    grad2 = np.reshape(grad[12:], (4, 5))
                    grad = np.concatenate([grad1.ravel('F'), grad2.ravel('F')])
                    res = np.hstack([J, grad]).tolist()
                elif part_id == 5:
                    J, grad = func(self.t, 2, 4, 4, self.Xm, self.ym, 1.5)
                    grad1 = np.reshape(grad[:12], (4, 3))
                    grad2 = np.reshape(grad[12:], (4, 5))
                    grad = np.concatenate([grad1.ravel('F'), grad2.ravel('F')])
                    res = np.hstack([J, grad]).tolist()
                else:
                    raise KeyError
                yield part_id, res
            except KeyError:
                yield part_id, 0


def sigmoid_gradient(z):
    """
    Computes the gradient of the sigmoid function evaluated at z.
    This should work regardless if z is a matrix or a vector.
    In particular, if z is a vector or matrix, you should return
    the gradient for each element.

    Parameters
    ----------
    z : array_like
        A vector or matrix as input to the sigmoid function.

    Returns
    --------
    g : array_like
        Gradient of the sigmoid function. Has the same shape as z.

    Instructions
    ------------
    Compute the gradient of the sigmoid function evaluated at
    each value of z (z can be a matrix, vector or scalar).

    Note
    ----
    Ypu can reuse the sigmoid function implemented
    in `logreg.py`.
    """

    g = np.zeros(z.shape)

    # ====================== YOUR CODE HERE ======================
    g = sigmoid(z) * (1 - sigmoid(z))
    # =============================================================
    return g


def nn_cost_function(nn_params,
                     input_layer_size,
                     hidden_layer_size,
                     num_labels,
                     X, y, lambda_=0.0):
    """
    Implements the neural network cost function and gradient for a two layer neural
    network which performs classification.

    Parameters
    ----------
    nn_params : array_like
        The parameters for the neural network which are "unrolled" into
        a vector. This needs to be converted back into the weight matrices theta1
        and theta2.

    input_layer_size : int
        Number of features for the input layer.

    hidden_layer_size : int
        Number of hidden units in the second layer.

    num_labels : int
        Total number of labels, or equivalently number of units in output layer.

    X : array_like
        Input dataset. A matrix of shape (m x input_layer_size).

    y : array_like
        Dataset labels. A vector of shape (m,).

    lambda_ : float, optional
        Regularization parameter.

    Returns
    -------
    J : float
        The computed value for the cost function at the current weight values.

    grad : array_like
        An "unrolled" vector of the partial derivatives of the concatenatation of
        neural network weights theta1 and theta2.

    Instructions
    ------------
    You should complete the code by working through the following parts.

    - Part 1: Feedforward the neural network and return the cost in the
              variable J.

    - Part 2: Implement the backpropagation algorithm to compute the gradients
              theta1_grad and theta2_grad. You should return the partial derivatives of
              the cost function with respect to theta1 and theta2 in theta1_grad and
              theta2_grad, respectively. After implementing Part 2, you can check
              that your implementation is correct by running checkNNGradients provided
              in the utils.py module.

              Note: The vector y passed into the function is a vector of labels
                    containing values from 0..K-1. You need to map this vector into a
                    binary vector of 1's and 0's to be used with the neural network
                    cost function.

              Hint: We recommend implementing backpropagation using a for-loop
                    over the training examples if you are implementing it for the
                    first time.

    - Part 3: Implement regularization with the cost function and gradients.

              Hint: You can implement this around the code for
                    backpropagation. That is, you can compute the gradients for
                    the regularization separately and then add them to theta1_grad
                    and theta2_grad from Part 2.

    Note
    ----
    We have provided an implementation for the sigmoid function in the file
    `utils.py` accompanying this assignment.
    """
    # Reshape nn_params back into the parameters theta1 and theta2, the weight matrices
    # for our 2 layer neural network
    theta1 = np.reshape(nn_params[:hidden_layer_size * (input_layer_size + 1)],
                        (hidden_layer_size, (input_layer_size + 1)))

    theta2 = np.reshape(nn_params[(hidden_layer_size * (input_layer_size + 1)):],
                        (num_labels, (hidden_layer_size + 1)))

    # Setup some useful variables
    m = y.size

    # You need to return the following variables correctly
    J = 0
    theta1_grad = np.zeros(theta1.shape)
    theta2_grad = np.zeros(theta2.shape)

    # ====================== YOUR CODE HERE ======================

    y_matrix = y.reshape(-1)
    y_matrix = np.eye(num_labels)[y_matrix]

    a1, a2, a3 = forward_propagation(theta1, theta2, X)
    J = (1 / m) * np.sum(-(y_matrix * np.log(a3)) - ((1 - y_matrix) * np.log(1 - a3)))

    theta1_reg = theta1.copy()  # (25, 401)
    theta1_reg[:, 0] = np.zeros(theta1_reg.shape[0])

    theta2_reg = theta2.copy()  # (10, 26)
    theta2_reg[:, 0] = np.zeros(theta2_reg.shape[0])

    J += (lambda_ / (2 * m)) * (np.sum(np.square(theta1_reg)) + np.sum(np.square(theta2_reg)))

    # backpropagation
    delta_3 = a3 - y_matrix
    z2 = np.dot(a1, theta1.T)
    delta_2 = np.dot(delta_3, theta2)[:, 1:] * sigmoid_gradient(z2)

    Delta_1 = np.dot(delta_2.T, a1)
    Delta_2 = np.dot(delta_3.T, a2)

    theta1_grad = (1 / m) * Delta_1
    theta1_grad[:, 1:] += (lambda_ / m) * theta1[:, 1:]
    theta2_grad = (1 / m) * Delta_2
    theta2_grad[:, 1:] += (lambda_ / m) * theta2[:, 1:]

    # ================================================================
    # Unroll gradients
    grad = np.concatenate([theta1_grad.ravel(), theta2_grad.ravel()])

    return J, grad


def rand_initialize_weights(L_in, L_out, epsilon_init=0.12):
    """
    Randomly initialize the weights of a layer in a neural network.

    Parameters
    ----------
    L_in : int
        Number of incomming connections.

    L_out : int
        Number of outgoing connections.

    epsilon_init : float, optional
        Range of values which the weight can take from a uniform
        distribution.

    Returns
    -------
    W : array_like
        The weight initialiatized to random values.  Note that W should
        be set to a matrix of size(L_out, 1 + L_in) as
        the first column of W handles the "bias" terms.

    Instructions
    ------------
    Initialize W randomly so that we break the symmetry while training
    the neural network. Note that the first column of W corresponds
    to the parameters for the bias unit.
    """

    # You need to return the following variables correctly
    W = np.zeros((L_out, 1 + L_in))

    # ====================== YOUR CODE HERE ======================

    W = np.random.rand(L_out, 1 + L_in) * 2 * epsilon_init - epsilon_init

    # ============================================================
    return W


def compute_numerical_gradient(J, theta, e=1e-4):
    """
    Computes the gradient using "finite differences" and gives us a numerical estimate of the
    gradient.

    Parameters
    ----------
    J : func
        The cost function which will be used to estimate its numerical gradient.

    theta : array_like
        The one dimensional unrolled network parameters. The numerical gradient is computed at
         those given parameters.

    e : float (optional)
        The value to use for epsilon for computing the finite difference.

    Notes
    -----
    The following code implements numerical gradient checking, and
    returns the numerical gradient. It sets `num_grad[i]` to (a numerical
    approximation of) the partial derivative of J with respect to the
    i-th input argument, evaluated at theta. (i.e., `num_grad[i]` should
    be the (approximately) the partial derivative of J with respect
    to theta[i].)
    """
    num_grad = np.zeros(theta.shape)
    perturb = np.diag(e * np.ones(theta.shape))
    for i in range(theta.size):
        loss1, _ = J(theta - perturb[:, i])
        loss2, _ = J(theta + perturb[:, i])
        num_grad[i] = (loss2 - loss1) / (2 * e)
    return num_grad


def forward_propagation(Theta1, Theta2, X):
    a1 = X  # (m, 401)
    a1 = add_intercept(a1)

    z2 = np.dot(a1, Theta1.T)
    a2 = sigmoid(z2)  # (m, 25)

    a2 = add_intercept(a2)

    z3 = np.dot(a2, Theta2.T)
    a3 = sigmoid(z3)  # (m, 10)

    return a1, a2, a3


def predict(Theta1, Theta2, X):
    """
    Predict the label of an input given a trained neural network
    Outputs the predicted label of X given the trained weights of a neural
    network(Theta1, Theta2)
    """

    a1, a2, a3 = forward_propagation(Theta1, Theta2, X)
    p = np.argmax(a3, axis=1)

    return p


def debug_initialize_weights(fan_out, fan_in):
    """
    Initialize the weights of a layer with fan_in incoming connections and fan_out outgoings
    connections using a fixed strategy. This will help you later in debugging.

    Note that W should be set a matrix of size (1+fan_in, fan_out) as the first row of W handles
    the "bias" terms.

    Parameters
    ----------
    fan_out : int
        The number of outgoing connections.

    fan_in : int
        The number of incoming connections.

    Returns
    -------
    W : array_like (1+fan_in, fan_out)
        The initialized weights array given the dimensions.
    """
    # Initialize W using "sin". This ensures that W is always of the same values and will be
    # useful for debugging
    W = np.sin(np.arange(1, 1 + (1 + fan_in) * fan_out)) / 10.0
    W = W.reshape(fan_out, 1 + fan_in, order='F')
    return W


def check_nn_gradients(nn_cost_function, lambda_=0):
    """
    Creates a small neural network to check the backpropagation gradients. It will output the
    analytical gradients produced by your backprop code and the numerical gradients
    (computed using compute_numerical_gradient). These two gradient computations should result in
    very similar values.

    Parameters
    ----------
    nn_cost_function : func
        A reference to the cost function implemented by the student.

    lambda_ : float (optional)
        The regularization parameter value.
    """
    input_layer_size = 3
    hidden_layer_size = 5
    num_labels = 3
    m = 5

    # We generate some 'random' test data
    Theta1 = debug_initialize_weights(hidden_layer_size, input_layer_size)
    Theta2 = debug_initialize_weights(num_labels, hidden_layer_size)

    # Reusing debug_initialize_weights to generate X
    X = debug_initialize_weights(m, input_layer_size - 1)
    y = np.arange(1, 1 + m) % num_labels
    # print(y)
    # Unroll parameters
    nn_params = np.concatenate([Theta1.ravel(), Theta2.ravel()])

    # short hand for cost function
    cost_func = lambda p: nn_cost_function(p, input_layer_size, hidden_layer_size,
                                           num_labels, X, y, lambda_)
    cost, grad = cost_func(nn_params)
    num_grad = compute_numerical_gradient(cost_func, nn_params)

    # Visually examine the two gradient computations.The two columns you get should be very similar.
    print(np.stack([num_grad, grad], axis=1))
    print('The above two columns you get should be very similar.')
    print('(Left-Your Numerical Gradient, Right-Analytical Gradient)\n')

    # Evaluate the norm of the difference between two the solutions. If you have a correct
    # implementation, and assuming you used e = 0.0001 in compute_numerical_gradient, then diff
    # should be less than 1e-9.
    diff = np.linalg.norm(num_grad - grad) / np.linalg.norm(num_grad + grad)

    print('If your backpropagation implementation is correct, then \n'
          'the relative difference will be small (less than 1e-9). \n'
          'Relative Difference: %g' % diff)


def cost(theta):
    return 3 * np.power(theta, 4) + 4


if __name__ == '__main__':
    e = 0.01
    j1 = cost(1 + e)
    j2 = cost(1 - e)

    print((j1 - j2) / (2 * e))
