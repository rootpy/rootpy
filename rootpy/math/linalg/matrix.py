import ROOT


def Matrix(rows, cols, type='F'):

    if type == 'F':
        return ROOT.TMatrixF(rows, cols)
    elif type == 'D':
        return ROOT.TMatrixD(rows, cols)
    raise TypeError("No matrix for type '%s'" % type)
