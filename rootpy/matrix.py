from ROOT import TMatrixD, TMatrixF

def Matrix(rows, cols, type='F'):

    if type == 'F':
        return TMatrixF(rows, cols)
    elif type == 'D':
        return TMatrixD(rows, cols)
    raise TypeError("No matrix for type '%s'" % type)
