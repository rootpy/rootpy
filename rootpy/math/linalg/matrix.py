# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License

import numpy as np
import ROOT


def Matrix(rows, cols, type='F'):

    if type == 'F':
        return ROOT.TMatrixF(rows, cols)
    elif type == 'D':
        return ROOT.TMatrixD(rows, cols)
    raise TypeError("No matrix for type '%s'" % type)

def as_numpy(root_matrix):
    """
    Returns the given ``root_matrix`` as a ``numpy.matrix``.
    """
    cols, rows = root_matrix.GetNcols(), root_matrix.GetNrows()
    return np.matrix([[root_matrix[i][j] for j in range(cols)]
                      for i in xrange(rows)])
