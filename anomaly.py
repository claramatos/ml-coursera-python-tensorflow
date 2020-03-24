import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from submission import Submission


class AnomalyGrader(Submission):
    # Random Test Cases
    n_u = 3
    n_m = 4
    n = 5
    X = np.sin(np.arange(1, 1 + n_m * n)).reshape(n_m, n, order='F')
    Theta = np.cos(np.arange(1, 1 + n_u * n)).reshape(n_u, n, order='F')
    Y = np.sin(np.arange(1, 1 + 2 * n_m * n_u, 2)).reshape(n_m, n_u, order='F')
    R = Y > 0.5
    pval = np.concatenate([abs(Y.ravel('F')), [0.001], [1]])
    Y = Y * R  # set 'Y' values to 0 for movies not reviewed

    yval = np.concatenate([R.ravel('F'), [1], [0]])
    #
    params = np.concatenate([X.ravel(), Theta.ravel()])

    def __init__(self):
        part_names = ['Estimate Gaussian Parameters',
                      'Select Threshold',
                      'Collaborative Filtering Cost',
                      'Collaborative Filtering Gradient',
                      'Regularized Cost',
                      'Regularized Gradient']
        super().__init__('anomaly-detection-and-recommender-systems', part_names)

    def __iter__(self):
        for part_id in range(1, 7):
            try:
                func = self.functions[part_id]

                # Each part has different expected arguments/different function
                if part_id == 1:
                    res = np.hstack(func(self.X)).tolist()
                elif part_id == 2:
                    res = np.hstack(func(self.yval, self.pval)).tolist()
                elif part_id == 3:
                    J, grad = func(self.params, self.Y, self.R, self.n_u, self.n_m, self.n)
                    res = J
                elif part_id == 4:
                    J, grad = func(self.params, self.Y, self.R, self.n_u, self.n_m, self.n, 0)
                    xgrad = grad[:self.n_m * self.n].reshape(self.n_m, self.n)
                    thetagrad = grad[self.n_m * self.n:].reshape(self.n_u, self.n)
                    res = np.hstack([xgrad.ravel('F'), thetagrad.ravel('F')]).tolist()
                elif part_id == 5:
                    res, _ = func(self.params, self.Y, self.R, self.n_u, self.n_m, self.n, 1.5)
                elif part_id == 6:
                    J, grad = func(self.params, self.Y, self.R, self.n_u, self.n_m, self.n, 1.5)
                    xgrad = grad[:self.n_m * self.n].reshape(self.n_m, self.n)
                    thetagrad = grad[self.n_m * self.n:].reshape(self.n_u, self.n)
                    res = np.hstack([xgrad.ravel('F'), thetagrad.ravel('F')]).tolist()
                else:
                    raise KeyError
                yield part_id, res
            except KeyError:
                yield part_id, 0


def normalize_ratings(Y, R):
    """
    Preprocess data by subtracting mean rating for every movie (every row).

    Parameters
    ----------
    Y : array_like
        The user ratings for all movies. A matrix of shape (num_movies x num_users).

    R : array_like
        Indicator matrix for movies rated by users. A matrix of shape (num_movies x num_users).

    Returns
    -------
    Ynorm : array_like
        A matrix of same shape as Y, after mean normalization.

    Ymean : array_like
        A vector of shape (num_movies, ) containing the mean rating for each movie.
    """
    m, n = Y.shape
    Ymean = np.zeros(m)
    Ynorm = np.zeros(Y.shape)

    for i in range(m):
        idx = R[i, :] == 1
        Ymean[i] = np.mean(Y[i, idx])
        Ynorm[i, idx] = Y[i, idx] - Ymean[i]

    return Ynorm, Ymean


def load_movie_list(filename):
    """
    Reads the fixed movie list in movie_ids.txt and returns a list of movie names.

    Returns
    -------
    movie_names : list
        A list of strings, representing all movie names.
    """
    # Read the fixed movieulary list
    with open(filename, encoding='ISO-8859-1') as fid:
        movies = fid.readlines()

    movie_names = []
    for movie in movies:
        parts = movie.split()
        movie_names.append(' '.join(parts[1:]).strip())
    return movie_names


def compute_numerical_gradient(J, theta, e=1e-4):
    """
    Computes the gradient using "finite differences" and gives us a
    numerical estimate of the gradient.

    Parameters
    ----------
    J : func
        The cost function which will be used to estimate its numerical gradient.

    theta : array_like
        The one dimensional unrolled network parameters. The numerical gradient
        is computed at those given parameters.

    e : float (optional)
        The value to use for epsilon for computing the finite difference.

    Returns
    -------
    numgrad : array_like
        The numerical gradient with respect to theta. Has same shape as theta.

    Notes
    -----
    The following code implements numerical gradient checking, and
    returns the numerical gradient. It sets `numgrad[i]` to (a numerical
    approximation of) the partial derivative of J with respect to the
    i-th input argument, evaluated at theta. (i.e., `numgrad[i]` should
    be the (approximately) the partial derivative of J with respect
    to theta[i].)
    """
    numgrad = np.zeros(theta.shape)
    perturb = np.diag(e * np.ones(theta.shape))
    for i in range(theta.size):
        loss1, _ = J(theta - perturb[:, i])
        loss2, _ = J(theta + perturb[:, i])
        numgrad[i] = (loss2 - loss1) / (2 * e)
    return numgrad


def check_cost_function(cofi_cost_func, lambda_=0.):
    """
    Creates a collaborative filtering problem to check your cost function and gradients.
    It will output the  analytical gradients produced by your code and the numerical gradients
    (computed using compute_numerical_gradient). These two gradient computations should result
    in very similar values.

    Parameters
    ----------
    cofi_cost_func: func
        Implementation of the cost function.

    lambda_ : float, optional
        The regularization parameter.
    """
    # Create small problem
    X_t = np.random.rand(4, 3)
    Theta_t = np.random.rand(5, 3)

    # Zap out most entries
    Y = np.dot(X_t, Theta_t.T)
    Y[np.random.rand(*Y.shape) > 0.5] = 0
    R = np.zeros(Y.shape)
    R[Y != 0] = 1

    # Run Gradient Checking
    X = np.random.randn(*X_t.shape)
    Theta = np.random.randn(*Theta_t.shape)
    num_movies, num_users = Y.shape
    num_features = Theta_t.shape[1]

    params = np.concatenate([X.ravel(), Theta.ravel()])
    numgrad = compute_numerical_gradient(
        lambda x: cofi_cost_func(x, Y, R, num_users, num_movies, num_features, lambda_), params)

    cost, grad = cofi_cost_func(params, Y, R, num_users, num_movies, num_features, lambda_)

    print(np.stack([numgrad, grad], axis=1))
    print('\nThe above two columns you get should be very similar.'
          '(Left-Your Numerical Gradient, Right-Analytical Gradient)')

    diff = np.linalg.norm(numgrad - grad) / np.linalg.norm(numgrad + grad)
    print('If your cost function implementation is correct, then '
          'the relative difference will be small (less than 1e-9).')
    print('\nRelative Difference: %g' % diff)


def multivariate_gaussian(X, mu, sigma2):
    """
    Computes the probability density function of the multivariate gaussian distribution.

    Parameters
    ----------
    X : array_like
        The dataset of shape (m x n). Where there are m examples of n-dimensions.

    mu : array_like
        A vector of shape (n,) contains the means for each dimension (feature).

    sigma2 : array_like
        Either a vector of shape (n,) containing the variances of independent features
        (i.e. it is the diagonal of the correlation matrix), or the full
        correlation matrix of shape (n x n) which can represent dependent features.

    Returns
    ------
    p : array_like
        A vector of shape (m,) which contains the computed probabilities at each of the
        provided examples.
    """
    k = mu.size

    # if sigma is given as a diagonal, compute the matrix
    if sigma2.ndim == 1:
        sigma2 = np.diag(sigma2)

    X = X - mu
    p = (2 * np.pi) ** (- k / 2) * np.linalg.det(sigma2) ** (-0.5) \
        * np.exp(-0.5 * np.sum(np.dot(X, np.linalg.pinv(sigma2)) * X, axis=1))
    return p


def visualize_fit(X, mu, sigma2):
    """
    Visualize the dataset and its estimated distribution.
    This visualization shows you the  probability density function of the Gaussian distribution.
    Each example has a location (x1, x2) that depends on its feature values.

    Parameters
    ----------
    X : array_like
        The dataset of shape (m x 2). Where there are m examples of 2-dimensions. We need at most
        2-D features to be able to visualize the distribution.

    mu : array_like
        A vector of shape (n,) contains the means for each dimension (feature).

    sigma2 : array_like
        Either a vector of shape (n,) containing the variances of independent features
        (i.e. it is the diagonal of the correlation matrix), or the full
        correlation matrix of shape (n x n) which can represent dependent features.
    """

    X1, X2 = np.meshgrid(np.arange(0, 35.5, 0.5), np.arange(0, 35.5, 0.5))
    Z = multivariate_gaussian(np.stack([X1.ravel(), X2.ravel()], axis=1), mu, sigma2)
    Z = Z.reshape(X1.shape)

    plt.plot(X[:, 0], X[:, 1], 'bx', mec='b', mew=2, ms=8)

    if np.all(abs(Z) != np.inf):
        plt.contour(X1, X2, Z, levels=10 ** (np.arange(-20., 1, 3)), zorder=100)


def estimate_gaussian(X):
    """
    This function estimates the parameters of a Gaussian distribution
    using a provided dataset.

    Parameters
    ----------
    X : array_like
        The dataset of shape (m x n) with each n-dimensional
        data point in one row, and each total of m data points.

    Returns
    -------
    mu : array_like
        A vector of shape (n,) containing the means of each dimension.

    sigma2 : array_like
        A vector of shape (n,) containing the computed
        variances of each dimension.

    Instructions
    ------------
    Compute the mean of the data and the variances
    In particular, mu[i] should contain the mean of
    the data for the i-th feature and sigma2[i]
    should contain variance of the i-th feature.
    """
    # Useful variables
    m, n = X.shape

    # You should return these values correctly
    mu = np.zeros(n)
    sigma2 = np.zeros(n)

    # ====================== YOUR CODE HERE ======================
    mu = (1 / m) * tf.reduce_sum(X, axis=0)
    sigma2 = (1 / m) * tf.reduce_sum(tf.square(X - mu), axis=0)
    # =============================================================
    return mu.numpy(), sigma2.numpy()


def select_threshold(yval, pval):
    """
    Find the best threshold (epsilon) to use for selecting outliers based
    on the results from a validation set and the ground truth.

    Parameters
    ----------
    yval : array_like
        The ground truth labels of shape (m, ).

    pval : array_like
        The precomputed vector of probabilities based on mu and sigma2 parameters. It's shape is also (m, ).

    Returns
    -------
    best_epsilon : array_like
        A vector of shape (n,) corresponding to the threshold value.

    best_f1 : float
        The value for the best f1 score.

    Instructions
    ------------
    Compute the f1 score of choosing epsilon as the threshold and place the
    value in f1. The code at the end of the loop will compare the
    f1 score for this choice of epsilon and set it to be the best epsilon if
    it is better than the current choice of epsilon.

    Notes
    -----
    You can use predictions = (pval < epsilon) to get a binary vector
    of 0's and 1's of the outlier predictions
    """
    best_epsilon = 0
    best_f1 = 0
    f1 = 0

    for epsilon in np.linspace(1.01 * min(pval), max(pval), 1000):
        # ====================== YOUR CODE HERE =======================
        ypred = np.zeros(yval.shape[0])
        ypred[pval < epsilon] = 1

        tp = np.sum((ypred == 1) & (yval == 1))
        fp = np.sum((ypred == 1) & (yval == 0))
        fn = np.sum((ypred == 0) & (yval == 1))

        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        f1 = (2 * prec * rec) / (prec + rec)

        # =============================================================
        if f1 > best_f1:
            best_f1 = f1
            best_epsilon = epsilon

    return best_epsilon, best_f1


def cofi_cost_func(params, Y, R, num_users, num_movies,
                   num_features, lambda_=0.0):
    """
    Collaborative filtering cost function.

    Parameters
    ----------
    params : array_like
        The parameters which will be optimized. This is a one
        dimensional vector of shape (num_movies x num_users, 1). It is the
        concatenation of the feature vectors X and parameters Theta.

    Y : array_like
        A matrix of shape (num_movies x num_users) of user ratings of movies.

    R : array_like
        A (num_movies x num_users) matrix, where R[i, j] = 1 if the
        i-th movie was rated by the j-th user.

    num_users : int
        Total number of users.

    num_movies : int
        Total number of movies.

    num_features : int
        Number of features to learn.

    lambda_ : float, optional
        The regularization coefficient.

    Returns
    -------
    J : float
        The value of the cost function at the given params.

    grad : array_like
        The gradient vector of the cost function at the given params.
        grad has a shape (num_movies x num_users, 1)

    Instructions
    ------------
    Compute the cost function and gradient for collaborative filtering.
    Concretely, you should first implement the cost function (without
    regularization) and make sure it is matches our costs. After that,
    you should implement thegradient and use the checkCostFunction routine
    to check that the gradient is correct. Finally, you should implement
    regularization.

    Notes
    -----
    - The input params will be unraveled into the two matrices:
        X : (num_movies  x num_features) matrix of movie features
        Theta : (num_users  x num_features) matrix of user features

    - You should set the following variables correctly:

        X_grad : (num_movies x num_features) matrix, containing the
                 partial derivatives w.r.t. to each element of X
        Theta_grad : (num_users x num_features) matrix, containing the
                     partial derivatives w.r.t. to each element of Theta

    - The returned gradient will be the concatenation of the raveled
      gradients X_grad and Theta_grad.
    """
    # Unfold the U and W matrices from params
    X = params[:num_movies * num_features].reshape(num_movies, num_features)
    Theta = params[num_movies * num_features:].reshape(num_users, num_features)

    # You need to return the following values correctly
    J = 0
    X_grad = np.zeros(X.shape)
    Theta_grad = np.zeros(Theta.shape)

    nm, nu = R.shape

    # ====================== YOUR CODE HERE ======================
    J = 0.5 * tf.reduce_sum(tf.square(tf.tensordot(X, Theta.T, axes=1) - Y) * R)
    J += (lambda_ / 2) * tf.reduce_sum((tf.square(Theta)))
    J += (lambda_ / 2) * tf.reduce_sum((tf.square(X)))

    for i in range(nm):
        idx = np.where(R[i, :] == 1)[0]
        Theta_temp = Theta[idx, :]
        Y_temp = Y[i, idx]
        X_grad[i, :] = tf.tensordot(tf.tensordot(X[i, :], Theta_temp.T, axes=1) - Y_temp, Theta_temp, axes=1)
        X_grad[i, :] += lambda_ * X[i, :]

    for j in range(nu):
        idx = np.where(R[:, j] == 1)[0]
        X_temp = X[idx, :]
        Y_temp = Y[idx, j]
        Theta_grad[j, :] = tf.tensordot(tf.tensordot(X_temp, Theta[j, :], axes=1) - Y_temp, X_temp, axes=1)
        Theta_grad[j, :] += lambda_ * Theta[j, :]

    # =============================================================
    grad = np.concatenate([X_grad.ravel(), Theta_grad.ravel()])
    return J, grad
