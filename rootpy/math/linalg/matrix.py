# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
import ROOT


def Matrix(rows, cols, type='F'):

    if type == 'F':
        return ROOT.TMatrixF(rows, cols)
    elif type == 'D':
        return ROOT.TMatrixD(rows, cols)
    raise TypeError("No matrix for type '%s'" % type)
