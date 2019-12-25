import numpy as np
import matplotlib.pyplot as plt

from submission import Submission


class LogRegGrader(Submission):
    X = np.stack([np.ones(20),
                  np.exp(1) * np.sin(np.arange(1, 21)),
                  np.exp(0.5) * np.cos(np.arange(1, 21))], axis=1)

    y = (np.sin(X[:, 0] + X[:, 1]) > 0).astype(float)

    theta = np.array([0.25, 0.5, -0.5])

    lambda_ = 0.1

    def __init__(self):
        part_names = ['Sigmoid Function',
                      'Logistic Regression Cost',
                      'Logistic Regression Gradient',
                      'Predict',
                      'Regularized Logistic Regression Cost',
                      'Regularized Logistic Regression Gradient']
        super().__init__('logistic-regression', part_names)

    def __iter__(self):
        for part_id in range(1, 7):
            try:
                func = self.functions[part_id]

                # Each part has different expected arguments/different function
                if part_id == 1:
                    res = func(self.X)
                elif part_id == 2:
                    res = func(self.theta, self.X, self.y)
                elif part_id == 3:
                    res = func(self.theta, self.X, self.y)[1]
                elif part_id == 4:
                    res = func(self.theta, self.X)
                elif part_id == 5:
                    res = func(self.theta, self.X, self.y, self.lambda_)
                elif part_id == 6:
                    res = func(self.theta, self.X, self.y, self.lambda_)[1]
                else:
                    raise KeyError
                yield part_id, res
            except KeyError:
                yield part_id, 0


def plot_data(X, y, xlabel='', ylabel='', legend=None):
    """
    Plots the data points X and y into a new figure. Plots the data
    points with * for the positive examples and o for the negative examples.

    Parameters
    ----------
    X : array_like
        An Mx2 matrix representing the dataset.

    y : array_like
        Label values for the dataset. A vector of size (M, ).

    xlabel: string
        label of the x axis

    ylabel:
        label of the y axis

    legend:
        legend

    Instructions
    ------------
    Plot the positive and negative examples on a 2D plot, using the
    option 'k*' for the positive examples and 'ko' for the negative examples.
    """
    # Create New Figure
    plt.figure()

    # ====================== YOUR CODE HERE ======================
    # Find Indices of Positive and Negative Examples
    pos = y == 1
    neg = y == 0

    # Plot Examples
    plt.plot(X[pos, 0], X[pos, 1], 'k*', lw=2, ms=10)
    plt.plot(X[neg, 0], X[neg, 1], 'ko', mfc='y', ms=8, mec='k', mew=1)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if legend is not None:
        plt.legend(legend)

    # ============================================================


def sigmoid(z):
    """
    Compute sigmoid function given the input z.

    Parameters
    ----------
    z : array_like
        The input to the sigmoid function. This can be a 1-D vector
        or a 2-D matrix.

    Returns
    -------
    g : array_like
        The computed sigmoid function. g has the same shape as z, since
        the sigmoid is computed element-wise on z.

    Instructions
    ------------
    Compute the sigmoid of each value of z (z can be a matrix, vector or scalar).
    """
    # convert input to a numpy array
    z = np.array(z)

    # You need to return the following variables correctly
    g = np.zeros(z.shape)

    # ====================== YOUR CODE HERE ======================
    g = 1 / (1 + np.exp(-z))
    # =============================================================
    return g


def hypothesis(X, theta):
    """
    Hypothesis function for linear regression.

    :param x: array_like
        Feature vector of shape (m, n+1), where m is the number of training
        examples and n is the number of features.
    :param theta: array_like
        Weight vector.

    :return:
    """
    return np.dot(X, theta)


def cost_function(theta, X, y):
    """
    Compute cost and gradient for logistic regression.

    Parameters
    ----------
    theta : array_like
        The parameters for logistic regression. This a vector
        of shape (n+1, ).

    X : array_like
        The input dataset of shape (m x n+1) where m is the total number
        of data points and n is the number of features. We assume the
        intercept has already been added to the input.

    y : arra_like
        Labels for the input. This is a vector of shape (m, ).

    Returns
    -------
    J : float
        The computed value for the cost function.

    grad : array_like
        A vector of shape (n+1, ) which is the gradient of the cost
        function with respect to theta, at the current values of theta.

    Instructions
    ------------
    Compute the cost of a particular choice of theta. You should set J to
    the cost. Compute the partial derivatives and set grad to the partial
    derivatives of the cost w.r.t. each parameter in theta.
    """
    # Initialize some useful values
    m = y.size  # number of training examples

    # You need to return the following variables correctly
    J = 0
    grad = np.zeros(theta.shape)

    # ====================== YOUR CODE HERE ======================
    z = hypothesis(X, theta)

    h = sigmoid(z)

    J = (1 / m) * (-np.dot(y, np.log(h)) - np.dot((1 - y), np.log(1 - h)))

    grad = (1 / m) * np.dot(h - y, X)

    # =============================================================
    return J, grad


def predict(theta, X):
    """
    Predict whether the label is 0 or 1 using learned logistic regression.
    Computes the predictions for X using a threshold at 0.5
    (i.e., if sigmoid(theta.T*x) >= 0.5, predict 1)

    Parameters
    ----------
    theta : array_like
        Parameters for logistic regression. A vecotor of shape (n+1, ).

    X : array_like
        The data to use for computing predictions. The rows is the number
        of points to compute predictions, and columns is the number of
        features.

    Returns
    -------
    p : array_like
        Predictions and 0 or 1 for each row in X.

    Instructions
    ------------
    Complete the following code to make predictions using your learned
    logistic regression parameters.You should set p to a vector of 0's and 1's
    """
    m = X.shape[0]  # Number of training examples

    # You need to return the following variables correctly
    p = np.zeros(m)

    # ====================== YOUR CODE HERE ======================

    z = hypothesis(X, theta)

    prob = sigmoid(z)

    p[prob > 0.5] = 1

    # ============================================================
    return p


def cost_function_reg(theta, X, y, lambda_):
    """
    Compute cost and gradient for logistic regression with regularization.

    Parameters
    ----------
    theta : array_like
        Logistic regression parameters. A vector with shape (n, ). n is
        the number of features including any intercept. If we have mapped
        our initial features into polynomial features, then n is the total
        number of polynomial features.

    X : array_like
        The data set with shape (m x n). m is the number of examples, and
        n is the number of features (after feature mapping).

    y : array_like
        The data labels. A vector with shape (m, ).

    lambda_ : float
        The regularization parameter.

    Returns
    -------
    J : float
        The computed value for the regularized cost function.

    grad : array_like
        A vector of shape (n, ) which is the gradient of the cost
        function with respect to theta, at the current values of theta.

    Instructions
    ------------
    Compute the cost `J` of a particular choice of theta.
    Compute the partial derivatives and set `grad` to the partial
    derivatives of the cost w.r.t. each parameter in theta.
    """
    # Initialize some useful values
    m = y.size  # number of training examples

    # You need to return the following variables correctly
    J = 0
    grad = np.zeros(theta.shape)

    # ===================== YOUR CODE HERE ======================

    J, grad = cost_function(theta, X, y)

    theta_reg = theta.copy()
    theta_reg[0] = 0

    J += (lambda_ / (2 * m)) * np.sum(np.square(theta_reg))

    grad += np.dot(lambda_ / m, theta_reg)

    # =============================================================
    return J, grad


if __name__ == '__main__':
    grader = LogRegGrader()
    grader[1] = sigmoid
    grader[2] = cost_function
    grader[3] = cost_function
    grader[4] = predict
    grader[5] = cost_function_reg
    grader[6] = cost_function_reg

    grader.grade()
