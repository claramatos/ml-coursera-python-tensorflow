import typing
import numpy as np
import matplotlib.pyplot as plt

from submission import Submission


class LinRegGrader(Submission):
    X1 = np.column_stack((np.ones(20), np.exp(1) + np.exp(2) * np.linspace(0.1, 2, 20)))
    Y1 = X1[:, 1] + np.sin(X1[:, 0]) + np.cos(X1[:, 1])
    X2 = np.column_stack((X1, X1[:, 1] ** 0.5, X1[:, 1] ** 0.25))
    Y2 = np.power(Y1, 0.5) + Y1
    theta1 = np.array([0.5, -0.5])
    theta2 = np.array([0.1, 0.2, 0.3, 0.4])
    theta3 = np.array([-0.1, -0.2, -0.3, -0.4])
    alpha = 0.01
    num_iters = 10

    def __init__(self):
        part_names = ['Warm up exercise',
                      'Computing Cost (for one variable)',
                      'Gradient Descent (for one variable)',
                      'Feature Normalization',
                      'Computing Cost (for multiple variables)',
                      'Gradient Descent (for multiple variables)',
                      'Normal Equations']
        super().__init__('linear-regression', part_names)

    def __iter__(self):
        for part_id in range(1, len(self.part_names) + 1):
            try:
                func = self.functions[part_id]

                # Each part has different expected arguments/different function
                if part_id == 1:
                    res = func()
                elif part_id == 2:
                    res = func(self.X1, self.Y1, self.theta1)
                elif part_id == 3:
                    res = func(self.X1, self.Y1, self.theta1, self.alpha, self.num_iters)
                elif part_id == 4:
                    res = func(self.X2[:, 1:4])
                elif part_id == 5:
                    res = func(self.X2, self.Y2, self.theta2)
                elif part_id == 6:
                    res = func(self.X2, self.Y2, self.theta3, self.alpha, self.num_iters)
                elif part_id == 7:
                    res = func(self.X2, self.Y2)
                else:
                    raise KeyError
                yield part_id, res
            except KeyError:
                yield part_id, 0


def warm_up():
    """
    An example function that returns the 5x5 identity matrix.

    :return: 5x5 identity matrix
    """

    # ============= YOUR CODE HERE ==============
    # Instructions: Return the 5x5 identity matrix
    #               In octave, we return values by defining which variables
    #               represent the return values (at the top of the file)
    #               and then set them accordingly.

    return np.eye(5)


def plot_data(X: np.array,
              y: np.array,
              x_label: typing.Optional[str] = '',
              y_label: typing.Optional[str] = ''):
    """
    Plots the data points x and y into a new figure.

    :param x: array_like
        The horizontal coordinates of the data points.
    :param y: array_like
        The vertical coordinates of the data points.
    :param x_label:  string, optional
        The label for the horizontal axis.
    :param y_label:  string, optional
        The label for the vertical axis.
    :return:
    """

    # ====================== YOUR CODE HERE ======================
    # Instructions: Plot the training data into a figure using the
    #               "figure" and "plot" commands. Set the axes labels using
    #               the "xlabel" and "ylabel" commands. Assume the
    #               population and revenue data have been passed in
    #               as the x and y arguments of this function.
    #
    # Hint: You can use the 'rx' option with plot to have the markers
    #       appear as red crosses. Furthermore, you can make the
    #       markers larger by using plt.plot(..., 'rx', markersize=10)

    plt.figure()
    plt.plot(X, y, 'rx', markersize=10)
    plt.xlabel(x_label)
    plt.ylabel(y_label)


def hypothesis(X: np.array, theta: np.array):
    """
    Hypothesis function for linear regression.

    :param x: array_like
        Feature vector of shape (m, n+1), where m is the number of training
        examples and n is the number of features.
    :param theta: array_like
        Weight vector.

    :return: Hypothesis function value for linear regression
    """
    return np.dot(X, theta)


def compute_cost(X: np.array, y: np.array, theta: np.array):
    """
    Computes the cost of using theta as the parameter for
    linear regression to fit the data points in x and y.

    :param x: array_like
        Feature vector of shape (m, n+1), where m is the number of training
        examples and n is the number of features.
    :param y: array_like
        Target vector of shape (m, ).
    :param theta: array_like
        Weight vector of shape (n+1, ).

    :return: cost value for the provided theta.
    """
    m = y.size

    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the cost of a particular choice of theta
    #               You should set J to the cost.
    h = hypothesis(X, theta)
    J = (1 / (2 * m)) * np.sum(np.square(h - y))

    # ==============================================================

    return J


def gradient_descent(X: np.array,
                     y: np.array,
                     theta: np.array,
                     alpha: float,
                     num_iters: int):
    """

    :param x: array_like
        Feature vector of shape (m, n+1), where m is the number of training
        examples and n is the number of features.
    :param y: array_like
        Target vector of shape (m, ).
    :param theta: array_like
        Weight vector of shape (n+1, ).
    :param alpha: scalar
        Learning rate.
    :param num_iters: scalar.
        Number of iterations.

    :return:
        theta : array_like
        The learned linear regression parameters. A vector of shape (n+1, ).

        j_history : list
        List for the values of the cost function after each iteration.
    """
    # Initialize some useful values
    m = y.size  # number of training examples
    j_history = np.zeros(num_iters)

    for iter in range(num_iters):
        # ====================== YOUR CODE HERE ======================
        # Instructions: Perform a single gradient step on the parameter vector
        #               theta.
        #
        # Hint: While debugging, it can be useful to print out the values
        #       of the cost function (compute_cost) and gradient here.
        #

        h = hypothesis(X, theta)
        theta -= (alpha / m) * np.dot(h - y, X)

        # ============================================================

        # Save the cost J in every iteration
        j_history[iter] = compute_cost(X, y, theta)

    return theta, j_history


def feature_normalize(X: np.array,
                      mu: np.array = None,
                      sigma: np.array = None):
    """
    Normalizes the features in x.

    :param x: array_like
        Feature vector.
    :param mu: scalar, optional
        Mean values of the feature vector.
    :param sigma:
        Standard deviation values of the feature vector.

    :return: returns a normalized version of X where the mean
    value of each feature is 0 and the standard deviation is 1.
    This is often a good preprocessing step to do when
    working with learning algorithms.
    """
    # ====================== YOUR CODE HERE ======================
    # Instructions: First, for each feature dimension, compute the mean
    #               of the feature and subtract it from the dataset,
    #               storing the mean value in mu. Next, compute the
    #               standard deviation of each feature and divide
    #               each feature by it's standard deviation, storing
    #               the standard deviation in sigma.
    #
    #               Note that X is a matrix where each column is a
    #               feature and each row is an example. You need
    #               to perform the normalization separately for
    #               each feature.
    #
    # Hint: You might find the 'np.mean' and 'np.std' functions useful.
    #

    if mu is None:
        mu = np.mean(X, axis=0)

    if sigma is None:
        sigma = np.std(X, axis=0)

    x_norm = (X - mu) / sigma

    # ============================================================

    return x_norm, mu, sigma


def compute_cost_multi(X: np.array, y: np.array, theta: np.array):
    """
    Compute cost for linear regression with multiple variables.
    Computes the cost of using theta as the parameter for linear
    regression to fit the data points in X and y.

    :param x: array_like
        Feature vector of shape (m, n+1), where m is the number of training
        examples and n is the number of features.
    :param y: array_like
        Target vector.
    :param theta: array_like
        Weight vector.

    :return: cost value for the provided theta.
    """
    return compute_cost(X, y, theta)


def gradient_descent_multi(X: np.array,
                           y: np.array,
                           theta: np.array,
                           alpha: float,
                           num_iters: int):
    """

    :param x: array_like
        Feature vector of shape (m, n+1), where m is the number of training
        examples and n is the number of features.
    :param y: array_like
        Target vector of shape (m, ).
    :param theta: array_like
        Weight vector of shape (n+1, ).
    :param alpha: scalar
        Learning rate.
    :param num_iters: scalar.
        Number of iterations.

    :return:
        theta : array_like
        The learned linear regression parameters. A vector of shape (n+1, ).

        J_history : list
        List for the values of the cost function after each iteration.
    """

    return gradient_descent(X, y, theta, alpha, num_iters)


def normal_eqn(X: np.array, y: np.array):
    """
    Computes the closed - form solution to linear regression
    using the normal equations.

    :param x: array_like
        Feature vector of shape (m, n+1), where m is the number of training
        examples and n is the number of features.
    :param y: array_like
        Target vector of shape (m, ).

    :return:
        theta : array_like
        The learned linear regression parameters. A vector of shape (n+1, ).
    """

    theta = np.zeros(X.shape[1])

    # ====================== YOUR CODE HERE ======================
    # Instructions: Complete the code to compute the closed form solution
    #               to linear regression and put the result in theta.
    #

    theta = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), y)

    # ============================================================

    return theta


if __name__ == "__main__":
    grader = LinRegGrader()
    grader[1] = warm_up
    grader[2] = compute_cost
    grader[3] = gradient_descent
    grader[4] = feature_normalize
    grader[5] = compute_cost_multi
    grader[6] = gradient_descent_multi
    grader[7] = normal_eqn

    grader.grade()
