# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
from __future__ import absolute_import
from ...extern.six.moves import range
from ...extern.six import string_types


__all__ = [
    'plot_corrcoef_matrix',
    'corrcoef',
    'cov',
]


def plot_corrcoef_matrix(matrix, names=None,
                         cmap=None, cmap_text=None,
                         fontsize=12, grid=False,
                         axes=None):
    """
    This function will draw a lower-triangular correlation matrix

    Parameters
    ----------

    matrix : 2-dimensional numpy array/matrix
        A correlation coefficient matrix

    names : list of strings, optional (default=None)
        List of the parameter names corresponding to the rows in ``matrix``.

    cmap : matplotlib color map, optional (default=None)
        Color map used to color the matrix cells.

    cmap_text : matplotlib color map, optional (default=None)
        Color map used to color the cell value text. If None, then
        all values will be black.

    fontsize : int, optional (default=12)
        Font size of parameter name and correlation value text.

    grid : bool, optional (default=False)
        If True, then draw dashed grid lines around the matrix elements.

    axes : matplotlib Axes instance, optional (default=None)
        The axes to plot on. If None then use the global current axes.

    Notes
    -----

    NumPy and matplotlib are required

    Examples
    --------

    >>> matrix = corrcoef(data.T, weights=weights)
    >>> plot_corrcoef_matrix(matrix, names)

    """
    import numpy as np
    from matplotlib import pyplot as plt
    from matplotlib import cm

    if axes is None:
        axes = plt.gca()

    matrix = np.asarray(matrix)

    if matrix.ndim != 2:
        raise ValueError("matrix is not a 2-dimensional array or matrix")
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix is not square")
    if names is not None and len(names) != matrix.shape[0]:
        raise ValueError("the number of names does not match the number of "
                         "rows/columns in the matrix")

    # mask out the upper triangular matrix
    matrix[np.triu_indices(matrix.shape[0])] = np.nan

    if isinstance(cmap_text, string_types):
        cmap_text = cm.get_cmap(cmap_text, 201)
    if cmap is None:
        cmap = cm.get_cmap('jet', 201)
    elif isinstance(cmap, string_types):
        cmap = cm.get_cmap(cmap, 201)
    # make NaN pixels white
    cmap.set_bad('w')

    axes.imshow(matrix, interpolation='nearest',
                cmap=cmap, origin='upper',
                vmin=-1, vmax=1)

    axes.set_frame_on(False)
    plt.setp(axes.get_yticklabels(), visible=False)
    plt.setp(axes.get_yticklines(), visible=False)
    plt.setp(axes.get_xticklabels(), visible=False)
    plt.setp(axes.get_xticklines(), visible=False)

    if grid:
        # draw grid lines
        for slot in range(1, matrix.shape[0] - 1):
            # vertical
            axes.plot((slot - 0.5, slot - 0.5),
                      (slot - 0.5, matrix.shape[0] - 0.5), 'k:', linewidth=1)
            # horizontal
            axes.plot((-0.5, slot + 0.5),
                      (slot + 0.5, slot + 0.5), 'k:', linewidth=1)
        if names is not None:
            for slot in range(1, matrix.shape[0]):
                # diagonal
                axes.plot((slot - 0.5, slot + 1.5),
                          (slot - 0.5, slot - 2.5), 'k:', linewidth=1)

    # label cell values
    for row, col in zip(*np.tril_indices(matrix.shape[0], k=-1)):
        value = matrix[row][col]
        if cmap_text is not None:
            color = cmap_text((value + 1.) / 2.)
        else:
            color = 'black'
        axes.text(
            col, row,
            "{0:d}%".format(int(value * 100)),
            color=color,
            ha='center', va='center',
            fontsize=fontsize)

    if names is not None:
        # write parameter names
        for i, name in enumerate(names):
            axes.annotate(
                name, (i, i),
                rotation=45,
                ha='left', va='bottom',
                transform=axes.transData,
                fontsize=fontsize)


def cov(m, y=None, rowvar=1, bias=0, ddof=None, weights=None, repeat_weights=0):
    """
    Estimate a covariance matrix, given data.

    Covariance indicates the level to which two variables vary together.
    If we examine N-dimensional samples, :math:`X = [x_1, x_2, ... x_N]^T`,
    then the covariance matrix element :math:`C_{ij}` is the covariance of
    :math:`x_i` and :math:`x_j`. The element :math:`C_{ii}` is the variance
    of :math:`x_i`.

    Parameters
    ----------
    m : array_like
        A 1-D or 2-D array containing multiple variables and observations.
        Each row of `m` represents a variable, and each column a single
        observation of all those variables. Also see `rowvar` below.
    y : array_like, optional
        An additional set of variables and observations. `y` has the same
        form as that of `m`.
    rowvar : int, optional
        If `rowvar` is non-zero (default), then each row represents a
        variable, with observations in the columns. Otherwise, the relationship
        is transposed: each column represents a variable, while the rows
        contain observations.
    bias : int, optional
        Default normalization is by ``(N - 1)``, where ``N`` is the number of
        observations given (unbiased estimate). If `bias` is 1, then
        normalization is by ``N``. These values can be overridden by using
        the keyword ``ddof`` in numpy versions >= 1.5.
    ddof : int, optional
        .. versionadded:: 1.5
        If not ``None`` normalization is by ``(N - ddof)``, where ``N`` is
        the number of observations; this overrides the value implied by
        ``bias``. The default value is ``None``.
    weights : array-like, optional
        A 1-D array of weights with a length equal to the number of
        observations.
    repeat_weights : int, optional
        The default treatment of weights in the weighted covariance is to first
        normalize them to unit sum and use the biased weighted covariance
        equation. If `repeat_weights` is 1 then the weights must represent an
        integer number of occurrences of each observation and both a biased and
        unbiased weighted covariance is defined because the total sample size
        can be determined.

    Returns
    -------
    out : ndarray
        The covariance matrix of the variables.

    See Also
    --------
    corrcoef : Normalized covariance matrix

    Examples
    --------
    Consider two variables, :math:`x_0` and :math:`x_1`, which
    correlate perfectly, but in opposite directions:

    >>> x = np.array([[0, 2], [1, 1], [2, 0]]).T
    >>> x
    array([[0, 1, 2],
           [2, 1, 0]])

    Note how :math:`x_0` increases while :math:`x_1` decreases. The covariance
    matrix shows this clearly:

    >>> np.cov(x)
    array([[ 1., -1.],
           [-1.,  1.]])

    Note that element :math:`C_{0,1}`, which shows the correlation between
    :math:`x_0` and :math:`x_1`, is negative.

    Further, note how `x` and `y` are combined:

    >>> x = [-2.1, -1,  4.3]
    >>> y = [3,  1.1,  0.12]
    >>> X = np.vstack((x,y))
    >>> print np.cov(X)
    [[ 11.71        -4.286     ]
     [ -4.286        2.14413333]]
    >>> print np.cov(x, y)
    [[ 11.71        -4.286     ]
     [ -4.286        2.14413333]]
    >>> print np.cov(x)
    11.71

    """
    import numpy as np
    # Check inputs
    if ddof is not None and ddof != int(ddof):
        raise ValueError(
            "ddof must be integer")

    X = np.array(m, ndmin=2, dtype=float)
    if X.size == 0:
        # handle empty arrays
        return np.array(m)
    if X.shape[0] == 1:
        rowvar = 1
    if rowvar:
        axis = 0
        tup = (slice(None), np.newaxis)
    else:
        axis = 1
        tup = (np.newaxis, slice(None))

    if y is not None:
        y = np.array(y, copy=False, ndmin=2, dtype=float)
        X = np.concatenate((X, y), axis)

    if ddof is None:
        if bias == 0:
            ddof = 1
        else:
            ddof = 0

    if weights is not None:
        weights = np.array(weights, dtype=float)
        weights_sum = weights.sum()
        if weights_sum <= 0:
            raise ValueError(
                "sum of weights is non-positive")
        X -= np.average(X, axis=1-axis, weights=weights)[tup]

        if repeat_weights:
            # each weight represents a number of repetitions of an observation
            # the total sample size can be determined in this case and we have
            # both an unbiased and biased weighted covariance
            fact = weights_sum - ddof
        else:
            # normalize weights so they sum to unity
            weights /= weights_sum
            # unbiased weighted covariance is not defined if the weights are
            # not integral frequencies (repeat-type)
            fact = (1. - np.power(weights, 2).sum())
    else:
        weights = 1
        X -= X.mean(axis=1-axis)[tup]
        if rowvar:
            N = X.shape[1]
        else:
            N = X.shape[0]
        fact = float(N - ddof)

    if not rowvar:
        return (np.dot(weights * X.T, X.conj()) / fact).squeeze()
    else:
        return (np.dot(weights * X, X.T.conj()) / fact).squeeze()


def corrcoef(x, y=None, rowvar=1, bias=0, ddof=None, weights=None,
             repeat_weights=0):
    """
    Return correlation coefficients.

    Please refer to the documentation for `cov` for more detail.  The
    relationship between the correlation coefficient matrix, `P`, and the
    covariance matrix, `C`, is

    .. math:: P_{ij} = \\frac{ C_{ij} } { \\sqrt{ C_{ii} * C_{jj} } }

    The values of `P` are between -1 and 1, inclusive.

    Parameters
    ----------
    x : array_like
        A 1-D or 2-D array containing multiple variables and observations.
        Each row of `m` represents a variable, and each column a single
        observation of all those variables. Also see `rowvar` below.
    y : array_like, optional
        An additional set of variables and observations. `y` has the same
        shape as `m`.
    rowvar : int, optional
        If `rowvar` is non-zero (default), then each row represents a
        variable, with observations in the columns. Otherwise, the relationship
        is transposed: each column represents a variable, while the rows
        contain observations.
    bias : int, optional
        Default normalization is by ``(N - 1)``, where ``N`` is the number of
        observations (unbiased estimate). If `bias` is 1, then
        normalization is by ``N``. These values can be overridden by using
        the keyword ``ddof`` in numpy versions >= 1.5.
    ddof : {None, int}, optional
        .. versionadded:: 1.5
        If not ``None`` normalization is by ``(N - ddof)``, where ``N`` is
        the number of observations; this overrides the value implied by
        ``bias``. The default value is ``None``.
    weights : array-like, optional
        A 1-D array of weights with a length equal to the number of
        observations.
    repeat_weights : int, optional
        The default treatment of weights in the weighted covariance is to first
        normalize them to unit sum and use the biased weighted covariance
        equation. If `repeat_weights` is 1 then the weights must represent an
        integer number of occurrences of each observation and both a biased and
        unbiased weighted covariance is defined because the total sample size
        can be determined.

    Returns
    -------
    out : ndarray
        The correlation coefficient matrix of the variables.

    See Also
    --------
    cov : Covariance matrix

    """
    import numpy as np
    c = cov(x, y, rowvar, bias, ddof, weights, repeat_weights)
    if c.size == 0:
        # handle empty arrays
        return c
    try:
        d = np.diag(c)
    except ValueError:  # scalar covariance
        return 1
    return c / np.sqrt(np.multiply.outer(d, d))
