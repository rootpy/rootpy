# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
from __future__ import absolute_import

from . import QROOT

__all__ = [
    'Matrix',
    'SymmetricMatrix',
]


class _MatrixBase(object):

    def __getitem__(self, loc):

        if isinstance(loc, tuple):
            i, j = loc
            return self(i, j)
        return super(_MatrixBase, self).__getitem__(loc)

    def __setitem__(self, loc, value):

        if isinstance(loc, tuple):
            i, j = loc
            # this is slow due to creation of temporaries
            self[i][j] = value
            return
        return super(_MatrixBase, self).__setitem__(loc, value)

    def to_numpy(self):
        """
        Convert this matrix into a ``numpy.matrix``.
        """
        import numpy as np
        cols, rows = self.GetNcols(), self.GetNrows()
        return np.matrix([[self(i, j)
            for j in range(cols)]
            for i in xrange(rows)])


class Matrix(object):

    @classmethod
    def dynamic_cls(cls, type='float'):

        class Matrix(_MatrixBase, QROOT.TMatrixT(type)):
            pass

        return Matrix

    def __new__(cls, *args, **kwargs):

        type = kwargs.pop('type', 'float')
        return cls.dynamic_cls(type)(*args, **kwargs)


class SymmetricMatrix(Matrix):

    @classmethod
    def dynamic_cls(cls, type='float'):

        class SymmetricMatrix(_MatrixBase, QROOT.TMatrixTSym(type)):
            pass

        return SymmetricMatrix
