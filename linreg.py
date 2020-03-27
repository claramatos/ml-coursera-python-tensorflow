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
    Example function in Python which computes the identity matrix.

    Returns
    -------
    A : array_like
        The 5x5 identity matrix.

    Instructions
    ------------
    Return the 5x5 identity matrix.
    """
    return np.eye(5)


def plot_data(X: np.array,
              y: np.array,
              x_label: typing.Optional[str] = '',
              y_label: typing.Optional[str] = ''):
    """
    Plots the data points x and y into a new figure. Plots the data
    points and gives the figure axes labels of population and profit.

    Parameters
    ----------
    x : array_like
        Data point values for x-axis.

    y : array_like
        Data point values for y-axis. Note x and y should have the same size.

    Instructions
    ------------
    Plot the training data into a figure using the "figure" and "plot"
    functions. Set the axes labels using the "xlabel" and "ylabel" functions.
    Assume the population and revenue data have been passed in as the x
    and y arguments of this function.

    Hint
    ----
    You can use the 'ro' option with plot to have the markers
    appear as red circles. Furthermore, you can make the markers larger by
    using plot(..., 'ro', ms=10), where `ms` refers to marker size. You
    can also set the marker edge color using the `mec` property.
    """
    plt.figure()  # open a new figure

    # ====================== YOUR CODE HERE =======================

    # =============================================================


def hypothesis(X: np.array, theta: np.array):
    """
    Hypothesis function for linear regression.

    Parameters
    ----------
    x: array_like
        Feature vector of shape (m, n+1), where m is the number of training
        examples and n is the number of features.
    theta: array_like
        Weight vector.

    Returns
    -------
    Hypothesis function value for linear regression
    """

    return np.dot(X, theta)


def compute_cost(X: np.array, y: np.array, theta: np.array):
    """
    Compute cost for linear regression. Computes the cost of using theta as the
    parameter for linear regression to fit the data points in X and y.

    Parameters
    ----------
    X : array_like
        The input dataset of shape (m x n+1), where m is the number of examples,
        and n is the number of features. We assume a vector of one's already
        appended to the features so we have n+1 columns.

    y : array_like
        The values of the function at each data point. This is a vector of
        shape (m, ).

    theta : array_like
        The parameters for the regression function. This is a vector of
        shape (n+1, ).

    Returns
    -------
    J : float
        The value of the regression cost function.

    Instructions
    ------------
    Compute the cost of a particular choice of theta.
    You should set J to the cost.
    """

    # number of training examples
    m = y.shape[0]
    J = np.zeros(m)

    # ====================== YOUR CODE HERE ======================

    # ==============================================================

    return J


def gradient_descent(X: np.array,
                     y: np.array,
                     theta: np.array,
                     alpha: float,
                     num_iters: int):
    """
    Performs gradient descent to learn `theta`. Updates theta by taking `num_iters`
    gradient steps with learning rate `alpha`.

    Parameters
    ----------
    X : array_like
        The input dataset of shape (m x n+1).

    y : arra_like
        Value at given features. A vector of shape (m, ).

    theta : array_like
        Initial values for the linear regression parameters.
        A vector of shape (n+1, ).

    alpha : float
        The learning rate.

    num_iters : int
        The number of iterations for gradient descent.

    Returns
    -------
    theta : array_like
        The learned linear regression parameters. A vector of shape (n+1, ).

    j_history : list
        A python list for the values of the cost function after each iteration.

    Instructions
    ------------
    Perform a single gradient step on the parameter vector theta.

    While debugging, it can be useful to print out the values of
    the cost function (computeCost) and gradient here.
    """
    # number of training examples
    m = y.size

    j_history = np.zeros(num_iters)

    for iter in range(num_iters):
        # ====================== YOUR CODE HERE ======================

        # ============================================================

        # Save the cost J in every iteration
        j_history[iter] = (compute_cost(X, y, theta))

    return theta, j_history


def feature_normalize(X: np.array,
                      mu: np.array = None,
                      sigma: np.array = None):
    """
    Normalizes the features in X. returns a normalized version of X where
    the mean value of each feature is 0 and the standard deviation
    is 1. This is often a good preprocessing step to do when working with
    learning algorithms.

    Parameters
    ----------
    X : array_like
        The dataset of shape (m x n).

    Returns
    -------
    X_norm : array_like
        The normalized dataset of shape (m x n).

    Instructions
    ------------
    First, for each feature dimension, compute the mean of the feature
    and subtract it from the dataset, storing the mean value in mu.
    Next, compute the  standard deviation of each feature and divide
    each feature by it's standard deviation, storing the standard deviation
    in sigma.

    Note that X is a matrix where each column is a feature and each row is
    an example. You needto perform the normalization separately for each feature.

    Hint
    ----
    You might find the 'np.mean' and 'np.std' functions useful.
    """
    x_norm = X.copy()
    mu = np.zeros(X.shape[1])
    sigma = np.zeros(X.shape[1])

    # ====================== YOUR CODE HERE ======================

    # ============================================================

    return x_norm, mu, sigma


def compute_cost_multi(X: np.array, y: np.array, theta: np.array):
    """
    Compute cost for linear regression with multiple variables.
    Computes the cost of using theta as the parameter for linear
    regression to fit the data points in X and y.

    Parameters
    ----------
    X : array_like
        The dataset of shape (m x n+1).

    y : array_like
        A vector of shape (m, ) for the values at a given data point.

    theta : array_like
        The linear regression parameters. A vector of shape (n+1, )

    Returns
    -------
    J : float
        The value of the cost function.

    Instructions
    ------------
    Compute the cost of a particular choice of theta. You should set J to the cost.
    """
    # number of training examples
    m = y.shape[0]

    # You need to return the following variable correctly
    J = 0

    # ======================= YOUR CODE HERE ===========================

    # ==================================================================
    return J


def gradient_descent_multi(X: np.array,
                           y: np.array,
                           theta: np.array,
                           alpha: float,
                           num_iters: int):
    """
    Performs gradient descent to learn theta.
    Updates theta by taking num_iters gradient steps with learning rate alpha.

    Parameters
    ----------
    X : array_like
        The dataset of shape (m x n+1).

    y : array_like
        A vector of shape (m, ) for the values at a given data point.

    theta : array_like
        The linear regression parameters. A vector of shape (n+1, )

    alpha : float
        The learning rate for gradient descent.

    num_iters : int
        The number of iterations to run gradient descent.

    Returns
    -------
    theta : array_like
        The learned linear regression parameters. A vector of shape (n+1, ).

    J_history : list
        A python list for the values of the cost function after each iteration.

    Instructions
    ------------
    Perform a single gradient step on the parameter vector theta.

    While debugging, it can be useful to print out the values of
    the cost function (computeCost) and gradient here.
    """

    # Initialize some useful values
    m = y.size  # number of training examples
    j_history = np.zeros(num_iters)

    for iter in range(num_iters):
        # ====================== YOUR CODE HERE ======================

        # ============================================================

        # Save the cost J in every iteration
        j_history[iter] = (compute_cost(X, y, theta))

    return theta, j_history


def normal_eqn(X: np.array, y: np.array):
    """
    Computes the closed-form solution to linear regression using the normal equations.

    Parameters
    ----------
    X : array_like
        The dataset of shape (m x n+1).

    y : array_like
        The value at each data point. A vector of shape (m, ).

    Returns
    -------
    theta : array_like
        Estimated linear regression parameters. A vector of shape (n+1, ).

    Instructions
    ------------
    Complete the code to compute the closed form solution to linear
    regression and put the result in theta.

    Hint
    ----
    Look up the function `np.linalg.pinv` for computing matrix inverse.
    """
    theta = np.zeros(X.shape[1])
    # ====================== YOUR CODE HERE ======================

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
