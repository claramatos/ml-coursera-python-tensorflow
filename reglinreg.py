import numpy as np
import matplotlib.pyplot as plt

from scipy import optimize

from submission import Submission
from utils import plot_fit


class RegLinRegGrader(Submission):
    # Random test cases
    X = np.vstack([np.ones(10),
                   np.sin(np.arange(1, 15, 1.5)),
                   np.cos(np.arange(1, 15, 1.5))]).T
    y = np.sin(np.arange(1, 31, 3))
    Xval = np.vstack([np.ones(10),
                      np.sin(np.arange(0, 14, 1.5)),
                      np.cos(np.arange(0, 14, 1.5))]).T
    yval = np.sin(np.arange(1, 11))

    def __init__(self):
        part_names = ['Regularized Linear Regression Cost Function',
                      'Regularized Linear Regression Gradient',
                      'Learning Curve',
                      'Polynomial Feature Mapping',
                      'Validation Curve']
        super().__init__('regularized-linear-regression-and-bias-variance', part_names)

    def __iter__(self):
        for part_id in range(1, 6):
            try:
                func = self.functions[part_id]
                # Each part has different expected arguments/different function
                if part_id == 1:
                    res = func(self.X, self.y, np.array([0.1, 0.2, 0.3]), 0.5)
                elif part_id == 2:
                    theta = np.array([0.1, 0.2, 0.3])
                    res = func(self.X, self.y, theta, 0.5)[1]
                elif part_id == 3:
                    res = np.hstack(func(self.X, self.y, self.Xval, self.yval, 1)).tolist()
                elif part_id == 4:
                    res = func(self.X[1, :].reshape(-1, 1), 8)
                elif part_id == 5:
                    res = np.hstack(func(self.X, self.y, self.Xval, self.yval)).tolist()
                else:
                    raise KeyError
            except KeyError:
                yield part_id, 0
            yield part_id, res


def linear_reg_cost_function(X, y, theta, lambda_=0.0):
    """
    Compute cost and gradient for regularized linear regression
    with multiple variables. Computes the cost of using theta as
    the parameter for linear regression to fit the data points in X and y.

    Parameters
    ----------
    X : array_like
        The dataset. Matrix with shape (m x n + 1) where m is the
        total number of examples, and n is the number of features
        before adding the bias term.

    y : array_like
        The functions values at each datapoint. A vector of
        shape (m, ).

    theta : array_like
        The parameters for linear regression. A vector of shape (n+1,).

    lambda_ : float, optional
        The regularization parameter.

    Returns
    -------
    J : float
        The computed cost function.

    grad : array_like
        The value of the cost function gradient w.r.t theta.
        A vector of shape (n+1, ).

    Instructions
    ------------
    Compute the cost and gradient of regularized linear regression for
    a particular choice of theta.
    You should set J to the cost and grad to the gradient.
    """
    # Initialize some useful values
    m = y.size  # number of training examples

    # You need to return the following variables correctly
    J = 0
    grad = np.zeros(theta.shape)

    # ====================== YOUR CODE HERE ======================

    # ============================================================
    return J, grad


def learning_curve(X, y, Xval, yval, lambda_=0):
    """
    Generates the train and cross validation set errors needed to plot a learning curve
    returns the train and cross validation set errors for a learning curve.

    In this function, you will compute the train and test errors for
    dataset sizes from 1 up to m. In practice, when working with larger
    datasets, you might want to do this in larger intervals.

    Parameters
    ----------
    X : array_like
        The training dataset. Matrix with shape (m x n + 1) where m is the
        total number of examples, and n is the number of features
        before adding the bias term.

    y : array_like
        The functions values at each training datapoint. A vector of
        shape (m, ).

    Xval : array_like
        The validation dataset. Matrix with shape (m_val x n + 1) where m is the
        total number of examples, and n is the number of features
        before adding the bias term.

    yval : array_like
        The functions values at each validation datapoint. A vector of
        shape (m_val, ).

    lambda_ : float, optional
        The regularization parameter.

    Returns
    -------
    error_train : array_like
        A vector of shape m. error_train[i] contains the training error for
        i examples.
    error_val : array_like
        A vecotr of shape m. error_val[i] contains the validation error for
        i training examples.

    Instructions
    ------------
    Fill in this function to return training errors in error_train and the
    cross validation errors in error_val. i.e., error_train[i] and
    error_val[i] should give you the errors obtained after training on i examples.

    Notes
    -----
    - You should evaluate the training error on the first i training
      examples (i.e., X[:i, :] and y[:i]).

      For the cross-validation error, you should instead evaluate on
      the _entire_ cross validation set (Xval and yval).

    - If you are using your cost function (linearRegCostFunction) to compute
      the training and cross validation error, you should call the function with
      the lambda argument set to 0. Do note that you will still need to use
      lambda when running the training to obtain the theta parameters.

    Hint
    ----
    You can loop over the examples with the following:

           for i in range(1, m+1):
               # Compute train/cross validation errors using training examples
               # X[:i, :] and y[:i], storing the result in
               # error_train[i-1] and error_val[i-1]
               ....
    """
    # Number of training examples
    m = y.size

    # You need to return these values correctly
    error_train = np.zeros(m)
    error_val = np.zeros(m)

    # ====================== YOUR CODE HERE ======================

    # ============================================================
    return error_train, error_val


def learning_curve_cv(X, y, Xval, yval, n, lambda_=0):
    # Number of training examples
    m = y.size
    mval = yval.size

    # You need to return these values correctly
    error_train = np.zeros((m, n))
    error_val = np.zeros((m, n))

    # ====================== YOUR CODE HERE ======================

    # =============================================================
    return error_train.mean(axis=1), error_val.mean(axis=1)


def poly_features(X, p):
    """
    Maps X (1D vector) into the p-th power.

    Parameters
    ----------
    X : array_like
        A data vector of size m, where m is the number of examples.

    p : int
        The polynomial power to map the features.

    Returns
    -------
    X_poly : array_like
        A matrix of shape (m x p) where p is the polynomial
        power and m is the number of examples. That is:

        X_poly[i, :] = [X[i], X[i]**2, X[i]**3 ...  X[i]**p]

    Instructions
    ------------
    Given a vector X, return a matrix X_poly where the p-th column of
    X contains the values of X to the p-th power.
    """
    # You need to return the following variables correctly.
    X_poly = np.zeros((X.shape[0], p))

    # ====================== YOUR CODE HERE ======================

    # ============================================================
    return X_poly


def validation_curve(X, y, Xval, yval):
    """
    Generate the train and validation errors needed to plot a validation
    curve that we can use to select lambda_.

    Parameters
    ----------
    X : array_like
        The training dataset. Matrix with shape (m x n) where m is the
        total number of training examples, and n is the number of features
        including any polynomial features.

    y : array_like
        The functions values at each training datapoint. A vector of
        shape (m, ).

    Xval : array_like
        The validation dataset. Matrix with shape (m_val x n) where m is the
        total number of validation examples, and n is the number of features
        including any polynomial features.

    yval : array_like
        The functions values at each validation datapoint. A vector of
        shape (m_val, ).

    Returns
    -------
    lambda_vec : list
        The values of the regularization parameters which were used in
        cross validation.

    error_train : list
        The training error computed at each value for the regularization
        parameter.

    error_val : list
        The validation error computed at each value for the regularization
        parameter.

    Instructions
    ------------
    Fill in this function to return training errors in `error_train` and
    the validation errors in `error_val`. The vector `lambda_vec` contains
    the different lambda parameters to use for each calculation of the
    errors, i.e, `error_train[i]`, and `error_val[i]` should give you the
    errors obtained after training with `lambda_ = lambda_vec[i]`.

    Note
    ----
    You can loop over lambda_vec with the following:

          for i in range(len(lambda_vec))
              lambda = lambda_vec[i]
              # Compute train / val errors when training linear
              # regression with regularization parameter lambda_
              # You should store the result in error_train[i]
              # and error_val[i]
              ....
    """
    # Selected values of lambda (you should not change this)
    lambda_vec = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]

    # You need to return these variables correctly.
    error_train = np.zeros(len(lambda_vec))
    error_val = np.zeros(len(lambda_vec))

    # ====================== YOUR CODE HERE ======================

    # ============================================================
    return lambda_vec, error_train, error_val


def train_linear_reg(linear_reg_cost_function, X, y, lambda_=0.0, maxiter=200):
    """
    Trains linear regression using scipy's optimize.minimize.

    Parameters
    ----------
    X : array_like
        The dataset with shape (m x n+1). The bias term is assumed to be concatenated.

    y : array_like
        Function values at each datapoint. A vector of shape (m,).

    lambda_ : float, optional
        The regularization parameter.

    maxiter : int, optional
        Maximum number of iteration for the optimization algorithm.

    Returns
    -------
    theta : array_like
        The parameters for linear regression. This is a vector of shape (n+1,).
    """
    # Initialize Theta
    initial_theta = np.zeros(X.shape[1])

    # Create "short hand" for the cost function to be minimized
    cost_function = lambda t: linear_reg_cost_function(X, y, t, lambda_)

    # Now, costFunction is a function that takes in only one argument
    options = {'maxiter': maxiter}

    # Minimize using scipy
    res = optimize.minimize(cost_function,
                            initial_theta,
                            jac=True,
                            method='TNC',
                            options=options)
    return res.x


def plot_poly_reg(X, y, X_poly, X_poly_val, yval, mu, sigma, theta, lambda_, p):
    m = X.shape[0]

    fig = plt.figure(figsize=(9, 4))

    plt.subplot(1, 2, 1)
    plt.plot(X, y, 'ro', ms=10, mew=1.5, mec='k')

    plot_fit(poly_features, np.min(X), np.max(X), mu, sigma, theta, p)

    plt.xlabel('Change in water level (x)')
    plt.ylabel('Water flowing out of the dam (y)')
    plt.title('Polynomial Regression \n Fit (lambda = %0.03f)' % lambda_)
    plt.ylim([-20, 50])

    plt.subplot(1, 2, 2)
    error_train, error_val = learning_curve(X_poly, y, X_poly_val, yval, lambda_)
    plt.plot(np.arange(1, 1 + m), error_train, np.arange(1, 1 + m), error_val)

    plt.title('Polynomial Regression \n Learning Curve (lambda = %0.03f)' % lambda_)
    plt.xlabel('Number of training examples')
    plt.ylabel('Error')
    plt.axis([0, 13, 0, 100])
    plt.legend(['Train', 'Cross Validation'])

    fig.tight_layout()

    print('Polynomial Regression (lambda = %f)\n' % lambda_)
    print('# Training Examples\tTrain Error\tCross Validation Error')
    for i in range(m):
        print('  \t%d\t\t%f\t%f' % (i + 1, error_train[i], error_val[i]))


if __name__ == '__main__':
    grader = RegLinRegGrader()
    grader[1] = linear_reg_cost_function
    grader[2] = linear_reg_cost_function
    grader[3] = learning_curve
    grader[4] = poly_features
    grader[5] = validation_curve

    grader.grade()
