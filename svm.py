import os
import re

import numpy as np

import matplotlib.pyplot as plt

from scipy.io import loadmat

from submission import Submission
from porterstemmer import PorterStemmer
from logreg import plot_data

RESOURCES_FOLDER = 'resources/svm/'


class SVMGrader(Submission):
    # Random Test Cases
    x1 = np.sin(np.arange(1, 11))
    x2 = np.cos(np.arange(1, 11))
    ec = 'the quick brown fox jumped over the lazy dog'
    wi = np.abs(np.round(x1 * 1863)).astype(int)
    wi = np.concatenate([wi, wi])
    vocab_list_filename = os.path.join(RESOURCES_FOLDER, 'vocab.txt')

    def __init__(self):
        part_names = ['Gaussian Kernel',
                      'Parameters (C, sigma) for Dataset 3',
                      'Email Processing',
                      'Email Feature Extraction']
        super().__init__('support-vector-machines', part_names)

    def __iter__(self):
        for part_id in range(1, 5):
            try:
                func = self.functions[part_id]
                # Each part has different expected arguments/different function
                if part_id == 1:
                    res = func(self.x1, self.x2, 2)
                elif part_id == 2:
                    res = np.hstack(func()).tolist()
                elif part_id == 3:
                    # add one to be compatible with matlab grader
                    res = [ind + 1 for ind in func(self.ec, self.vocab_list_filename, False)]
                elif part_id == 4:
                    res = func(self.wi)
                else:
                    raise KeyError
                yield part_id, res
            except KeyError:
                yield part_id, 0


def gaussian_kernel(x1, x2, sigma):
    """
    Computes the radial basis function
    Returns a radial basis function kernel between x1 and x2.

    Parameters
    ----------
    x1 :  numpy ndarray
        A vector of size (n, ), representing the first datapoint.

    x2 : numpy ndarray
        A vector of size (n, ), representing the second datapoint.

    sigma : float
        The bandwidth parameter for the Gaussian kernel.

    Returns
    -------
    sim : float
        The computed RBF between the two provided data points.

    Instructions
    ------------
    Fill in this function to return the similarity between `x1` and `x2`
    computed using a Gaussian kernel with bandwidth `sigma`.
    """
    sim = 0
    # ====================== YOUR CODE HERE ======================

    # ============================================================
    return sim


def linear_kernel(x1, x2):
    """
    Returns a linear kernel between x1 and x2.

    Parameters
    ----------
    x1 : numpy ndarray
        A 1-D vector.

    x2 : numpy ndarray
        A 1-D vector of same size as x1.

    Returns
    -------
    : float
        The scalar amplitude.
    """
    return np.dot(x1, x2)


def dataset3_params(X, y, Xval, yval):
    """
    Returns your choice of C and sigma for Part 3 of the exercise
    where you select the optimal (C, sigma) learning parameters to use for SVM
    with RBF kernel.

    Parameters
    ----------
    X : array_like
        (m x n) matrix of training data where m is number of training examples, and
        n is the number of features.

    y : array_like
        (m, ) vector of labels for ther training data.

    Xval : array_like
        (mv x n) matrix of validation data where mv is the number of validation examples
        and n is the number of features

    yval : array_like
        (mv, ) vector of labels for the validation data.

    Returns
    -------
    C, sigma : float, float
        The best performing values for the regularization parameter C and
        RBF parameter sigma.

    Instructions
    ------------
    Fill in this function to return the optimal C and sigma learning
    parameters found using the cross validation set.
    You can use `svmPredict` to predict the labels on the cross
    validation set. For example,

        predictions = svm_predict(model, Xval)

    will return the predictions on the cross validation set.

    Note
    ----
    You can compute the prediction error using

        np.mean(predictions != yval)
    """
    # You need to return the following variables correctly.
    C = 1
    sigma = 0.3

    # ====================== YOUR CODE HERE ======================

    # ============================================================
    return C, sigma


def process_email(email_contents, vocab_list_filename, verbose=True):
    """
    Preprocesses the body of an email and returns a list of indices
    of the words contained in the email.

    Parameters
    ----------
    email_contents : str
        A string containing one email.

    verbose : bool
        If True, print the resulting email after processing.

    Returns
    -------
    word_indices : list
        A list of integers containing the index of each word in the
        email which is also present in the vocabulary.

    Instructions
    ------------
    Fill in this function to add the index of word to word_indices
    if it is in the vocabulary. At this point of the code, you have
    a stemmed word from the email in the variable word.
    You should look up word in the vocabulary list (vocab_list).
    If a match exists, you should add the index of the word to the word_indices
    list. Concretely, if word = 'action', then you should
    look up the vocabulary list to find where in vocab_list
    'action' appears. For example, if vocab_list[18] =
    'action', then, you should add 18 to the word_indices
    vector (e.g., word_indices.append(18)).

    Notes
    -----
    - vocab_list[idx] returns a the word with index idx in the vocabulary list.

    - vocab_list.index(word) return index of word `word` in the vocabulary list.
      (A ValueError exception is raised if the word does not exist.)
    """
    # Load Vocabulary
    vocab_list = get_vocab_list(vocab_list_filename)

    # Init return value
    word_indices = []

    # ========================== Preprocess Email ===========================
    # Find the Headers ( \n\n and remove )
    # Uncomment the following lines if you are working with raw emails with the
    # full headers
    # hdrstart = email_contents.find(chr(10) + chr(10))
    # email_contents = email_contents[hdrstart:]

    # Lower case
    email_contents = email_contents.lower()

    # Strip all HTML
    # Looks for any expression that starts with < and ends with > and replace
    # and does not have any < or > in the tag it with a space
    email_contents = re.compile('<[^<>]+>').sub(' ', email_contents)

    # Handle Numbers
    # Look for one or more characters between 0-9
    email_contents = re.compile('[0-9]+').sub(' number ', email_contents)

    # Handle URLS
    # Look for strings starting with http:// or https://
    email_contents = re.compile('(http|https)://[^\s]*').sub(' httpaddr ', email_contents)

    # Handle Email Addresses
    # Look for strings with @ in the middle
    email_contents = re.compile('[^\s]+@[^\s]+').sub(' emailaddr ', email_contents)

    # Handle $ sign
    email_contents = re.compile('[$]+').sub(' dollar ', email_contents)

    # get rid of any punctuation
    email_contents = re.split('[ @$/#.-:&*+=\[\]?!(){},''">_<;%\n\r]', email_contents)

    # remove any empty word string
    email_contents = [word for word in email_contents if len(word) > 0]

    # Stem the email contents word by word
    stemmer = PorterStemmer()
    processed_email = []
    for word in email_contents:
        # Remove any remaining non alphanumeric characters in word
        word = re.compile('[^a-zA-Z0-9]').sub('', word).strip()
        word = stemmer.stem(word)
        processed_email.append(word)

        if len(word) < 1:
            continue

        # Look up the word in the dictionary and add to word_indices if found

        # ====================== YOUR CODE HERE ======================

        # =============================================================

    if verbose:
        print('----------------')
        print('Processed email:')
        print('----------------')
        print(' '.join(processed_email))
    return word_indices


def email_features(word_indices):
    """
    Takes in a word_indices vector and produces a feature vector from the word indices.

    Parameters
    ----------
    word_indices : list
        A list of word indices from the vocabulary list.

    Returns
    -------
    x : list
        The computed feature vector.

    Instructions
    ------------
    Fill in this function to return a feature vector for the
    given email (word_indices). To help make it easier to  process
    the emails, we have have already pre-processed each email and converted
    each word in the email into an index in a fixed dictionary (of 1899 words).
    The variable `word_indices` contains the list of indices of the words
    which occur in one email.

    Concretely, if an email has the text:

        The quick brown fox jumped over the lazy dog.

    Then, the word_indices vector for this text might look  like:

        60  100   33   44   10     53  60  58   5

    where, we have mapped each word onto a number, for example:

        the   -- 60
        quick -- 100
        ...

    Note
    ----
    The above numbers are just an example and are not the actual mappings.

    Your task is take one such `word_indices` vector and construct
    a binary feature vector that indicates whether a particular
    word occurs in the email. That is, x[i] = 1 when word i
    is present in the email. Concretely, if the word 'the' (say,
    index 60) appears in the email, then x[60] = 1. The feature
    vector should look like:
        x = [ 0 0 0 0 1 0 0 0 ... 0 0 0 0 1 ... 0 0 0 1 0 ..]
    """
    # Total number of words in the dictionary
    n = 1899

    # You need to return the following variables correctly.
    x = np.zeros(n)

    # ===================== YOUR CODE HERE ======================

    # ===========================================================

    return x


def gaussian_kernel_vec(X, sigma, Y=None):
    """
    Computes the vectorized radial basis function using
    the following equality:
    ||x-y||^2 = ||x||^2 + ||y||^2 - 2 * x^T * y

    Returns a radial basis function kernel between all
    the data points in x.

    Parameters
    ----------
    X :  numpy ndarray
        A vector of size (n, ), representing the array of data points.

    sigma : float
        The bandwidth parameter for the Gaussian kernel.

    Returns
    -------
    sim : float
        The computed RBF between the provided data points.
    """

    X1 = np.sum(X ** 2, axis=1)

    if Y is None:
        X2 = X1
        XT = X.T
    else:
        X2 = np.sum(Y ** 2, 1)
        XT = Y.T

    K = X2[None, :] + X1[:, None] - 2 * np.dot(X, XT)

    K /= 2 * sigma ** 2

    return np.exp(-K)


def non_linear_kernel(X, kernel_function):
    m, n = X.shape

    K = np.zeros((m, m))
    for i in range(m):
        for j in range(i, m):
            K[i, j] = kernel_function(X[i, :], X[j, :])
            K[j, i] = K[i, j]

    return K


_KERNEL = {
    'linear_kernel': linear_kernel,
    'gaussian_kernel': gaussian_kernel,
    'non_linear_kernel': non_linear_kernel,
}


def svm_kernel(X, kernel_function, args=()):
    # We have implemented the optimized vectorized version of the Kernels here so
    # that the SVM training will run faster
    if kernel_function.__name__ == 'linear_kernel':
        # Vectorized computation for the linear kernel
        # This is equivalent to computing the kernel on every pair of examples
        K = linear_kernel(X, X.T)
    elif kernel_function.__name__ == 'gaussian_kernel':
        # vectorized RBF Kernel: ||x-y||^2 = ||x||^2 + ||y||^2 - 2 * x^T * y
        # This is equivalent to computing the kernel on every pair of example)
        K = gaussian_kernel_vec(X, args[0])
    else:
        K = non_linear_kernel(X, kernel_function)

    return K


def svm_train(X, Y, C, kernel_function, tol=1e-3, max_passes=5, args=()):
    """
    Trains an SVM classifier using a  simplified version of the SMO algorithm.

    Parameters
    ---------
    X : numpy ndarray
        (m x n) Matrix of training examples. Each row is a training example, and the
        jth column holds the jth feature.

    Y : numpy ndarray
        (m, ) A vector (1-D numpy array) containing 1 for positive examples and 0 for negative examples.

    C : float
        The standard SVM regularization parameter.

    kernel_function : func
        A function handle which computes the kernel. The function should accept two vectors as
        inputs, and returns a scalar as output.

    tol : float, optional
        Tolerance value used for determining equality of floating point numbers.

    max_passes : int, optional
        Controls the number of iterations over the dataset (without changes to alpha)
        before the algorithm quits.

    args : tuple
        Extra arguments required for the kernel function, such as the sigma parameter for a
        Gaussian kernel.

    Returns
    -------
    model :
        The trained SVM model.

    Notes
    -----
    This is a simplified version of the SMO algorithm for training SVMs. In practice, if
    you want to train an SVM classifier, we recommend using an optimized package such as:

    - LIBSVM   (http://www.csie.ntu.edu.tw/~cjlin/libsvm/)
    - SVMLight (http://svmlight.joachims.org/)
    - scikit-learn (http://scikit-learn.org/stable/modules/svm.html) which contains python wrappers
    for the LIBSVM library.

    For a more complete description of the SMO algorithm see:
    http://cs229.stanford.edu/materials/smo.pdf
    """
    # make sure data is signed int
    Y = Y.astype(int)
    # Dataset size parameters
    m, n = X.shape

    passes = 0
    E = np.zeros(m)
    alphas = np.zeros(m)
    b = 0

    # Map 0 to -1
    Y[Y == 0] = -1

    # Pre-compute the Kernel Matrix since our dataset is small
    # (in practice, optimized SVM packages that handle large datasets
    # gracefully will **not** do this)
    K = svm_kernel(X, kernel_function, args)

    while passes < max_passes:
        num_changed_alphas = 0
        for i in range(m):
            # calculate Ei = f(x(i)) - y(i) using (2)
            E[i] = b + np.sum(alphas * Y * K[:, i]) - Y[i]

            if (Y[i] * E[i] < -tol and alphas[i] < C) or (Y[i] * E[i] > tol and alphas[i] > 0):

                # select the alpha_j randomly
                j = np.random.choice(list(range(i)) + list(range(i + 1, m)), size=1)[0]

                # calculate Ej = f(x(j)) - y(j) using (2)
                E[j] = b + np.sum(alphas * Y * K[:, j]) - Y[j]

                # save old alphas
                alpha_i_old = alphas[i]
                alpha_j_old = alphas[j]

                # compute L and H by (10) or (11)
                if Y[i] == Y[j]:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                else:
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])

                if L == H:
                    continue

                # compute eta by (14)
                eta = 2 * K[i, j] - K[i, i] - K[j, j]

                # objective function positive definite, there will be a minimum along the direction
                # of linear equality constraint, and eta will be greater than zero
                # we are actually computing -eta here (so we skip of eta >= 0)
                if eta >= 0:
                    continue

                # compute and clip new value for alpha j using (12) and (15)
                alphas[j] -= Y[j] * (E[i] - E[j]) / eta

                # clip
                alphas[j] = min(H, alphas[j])
                alphas[j] = max(L, alphas[j])

                # check if change in alpha is significant
                if abs(alphas[j] - alpha_j_old) < tol:
                    alphas[j] = alpha_j_old
                    continue

                # determine value for alpha i using (16)
                alphas[i] += Y[i] * Y[j] * (alpha_j_old - alphas[j])

                # compute b1 and b2 using (17) and (18) respectively
                b1 = b - E[i] - Y[i] * (alphas[i] - alpha_i_old) * K[i, j] \
                     - Y[j] * (alphas[j] - alpha_j_old) * K[i, j]

                b2 = b - E[j] - Y[i] * (alphas[i] - alpha_i_old) * K[i, j] \
                     - Y[j] * (alphas[j] - alpha_j_old) * K[j, j]

                # compute b by (19)
                if 0 < alphas[i] < C:
                    b = b1
                elif 0 < alphas[j] < C:
                    b = b2
                else:
                    b = (b1 + b2) / 2

                num_changed_alphas += 1

        if num_changed_alphas == 0:
            passes += 1
        else:
            passes = 0

    idx = alphas > 0
    model = {'X': X[idx, :],
             'y': Y[idx],
             'kernel_function': kernel_function,
             'b': b,
             'args': args,
             'alphas': alphas[idx],
             'w': np.dot(alphas * Y, X)}
    return model


def svm_predict(model, X):
    """
    Returns a vector of predictions using a trained SVM model.

    Parameters
    ----------
    model : dict
        The parameters of the trained svm model, as returned by the function svmTrain

    X : array_like
        A (m x n) matrix where each example is a row.

    Returns
    -------
    pred : array_like
        A (m,) sized vector of predictions {0, 1} values.
    """
    # check if we are getting a vector. If so, then assume we only need to do predictions
    # for a single example
    if X.ndim == 1:
        X = X[np.newaxis, :]

    m = X.shape[0]
    p = np.zeros(m)
    pred = np.zeros(m)

    if model['kernel_function'].__name__ == 'linear_kernel':
        # we can use the weights and bias directly if working with the linear kernel
        # p = w'*x + b
        p = np.dot(X, model['w']) + model['b']
    elif model['kernel_function'].__name__ == 'gaussian_kernel':
        # vectorized RBF Kernel
        # This is equivalent to computing the kernel on every pair of examples
        K = gaussian_kernel_vec(X, model['args'][0], model['X'])
        # p = sum(alpha * y * K + b)
        p = np.dot(K, model['alphas'] * model['y']) + model['b']
    else:
        # other non-linear kernel
        for i in range(m):
            predictions = 0
            for j in range(model['X'].shape[0]):
                predictions += model['alphas'][j] * model['y'][j] \
                               * model['kernel_function'](X[i, :], model['X'][j, :])
            p[i] = predictions

    pred[p >= 0] = 1

    return pred


def get_vocab_list(filename):
    """
    Reads the fixed vocabulary list in vocab.txt and returns a cell array of the words
    %   vocab_list = GETVOCABLIST() reads the fixed vocabulary list in vocab.txt
    %   and returns a cell array of the words in vocab_list.

    :return:
    """
    vocab_list = np.genfromtxt(filename, dtype=object)
    return list(vocab_list[:, 1].astype(str))


def visualize_boundary_linear(X, y, model):
    """
    Plots a linear decision boundary learned by the SVM.

    Parameters
    ----------
    X : array_like
        (m x 2) The training data with two features (to plot in a 2-D plane).

    y : array_like
        (m, ) The data labels.

    model : dict
        Dictionary of model variables learned by SVM.
    """
    w, b = model['w'], model['b']
    xp = np.linspace(min(X[:, 0]), max(X[:, 0]), 100)
    yp = -(w[0] * xp + b) / w[1]

    plot_data(X, y)
    plt.plot(xp, yp, '-b')


def visualize_boundary(X, y, model):
    """
    Plots a non-linear decision boundary learned by the SVM and overlays the data on it.

    Parameters
    ----------
    X : array_like
        (m x 2) The training data with two features (to plot in a 2-D plane).

    y : array_like
        (m, ) The data labels.

    model : dict
        Dictionary of model variables learned by SVM.
    """
    plot_data(X, y)

    # make classification predictions over a grid of values
    x1plot = np.linspace(min(X[:, 0]), max(X[:, 0]), 100)
    x2plot = np.linspace(min(X[:, 1]), max(X[:, 1]), 100)
    X1, X2 = np.meshgrid(x1plot, x2plot)

    vals = np.zeros(X1.shape)
    for i in range(X1.shape[1]):
        this_X = np.stack((X1[:, i], X2[:, i]), axis=1)
        vals[:, i] = svm_predict(model, this_X)

    plt.contour(X1, X2, vals, colors='y', linewidths=2)
    plt.pcolormesh(X1, X2, vals, cmap='YlGnBu', alpha=0.25, edgecolors='None', lw=0)
    plt.grid(False)


if __name__ == '__main__':
    data = loadmat(os.path.join(RESOURCES_FOLDER, 'ex6data3.mat'))
    X = data['X']
    y = data['y'][:, 0]
    Xval = data['Xval']
    yval = data['yval'][:, 0]
    C, sigma = dataset3_params(X, y, Xval, yval)

    grader = SVMGrader()
    grader[1] = gaussian_kernel
    grader[2] = lambda: (C, sigma)
    grader[3] = process_email
    grader[4] = email_features

    grader.grade()
