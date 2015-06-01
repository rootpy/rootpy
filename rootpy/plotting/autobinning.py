# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
from __future__ import absolute_import

import types
import numpy as np


__all__ = [
    'autobinning',
]


def autobinning(data, method="freedman_diaconis"):
    """
    This method determines the optimal binning for histogramming.

    Parameters
    ----------
    data: 1D array-like
          Input data.
    method: string, one of the following:
          - sturges
          - sturges-doane
          - scott
          - sqrt
          - doane
          - freedman-diaconis
          - risk
          - knuth

    Returns
    -------
    (nbins, min, max): int, type(data), type(data)
         nbins is the optimal number of bin estimated by the method
         min is the minimum of data
         max is the maximum of data

    Notes
    -----
    If the length of data is less than 4 the method retun nbins = 1
    """
    name = method.replace("-", "_")
    try:
        method = getattr(BinningMethods, name)
        if not isinstance(method, types.FunctionType):
            raise AttributeError
    except AttributeError:
        raise ValueError("`{0}` is not a valid binning method".format(name))
    if len(data) < 4:
        return 1, np.min(data), np.max(data)
    return int(np.ceil(method(data))), np.min(data), np.max(data)


class BinningMethods(object):
    """
    Static methods on this class are available as methods for ``autobinning``.
    """
    @classmethod
    def all_methods(cls):
        """
        Return the names of all available binning methods
        """
        def name(fn):
            return fn.__get__(cls).__name__.replace("_", "-")
        return sorted(name(f) for f in cls.__dict__.values()
                      if isinstance(f, staticmethod))

    @staticmethod
    def sturges(data):
        n = len(data)
        return np.log2(n) + 1

    @staticmethod
    def sturges_doane(data):
        """
        References
        ----------
        .. [1] D. Wilkinson, "The Grammar of Graphics", 2005.
               http://books.google.it/books?id=_kRX4LoFfGQC&lpg=PA133&ots=APHb0-p6tY&dq=doane%20binning%20histogram&hl=it&pg=PA133#v=onepage&q=doane%20binning%20histogram&f=false
        """
        n = len(data)
        return np.log10(n) * np.log2(n) + 3

    @staticmethod
    def doane(data):
        """
        Modified Doane modified
        """
        from scipy.stats import skew
        n = len(data)
        sigma = np.sqrt(6. * (n - 2.) / (n + 1.) / (n + 3.))
        return 1 + np.log2(n) + \
            np.log2(1 + np.abs(skew(data)) / sigma)

    @staticmethod
    def scott(data):
        sigma = np.std(data)
        n = len(data)
        h = 3.49 * sigma * n ** (-1. / 3.)
        return (np.max(data) - np.min(data)) / h

    @staticmethod
    def sqrt(data):
        return np.sqrt(len(data))

    @staticmethod
    def freedman_diaconis(data):
        from scipy.stats.mstats import mquantiles
        q = mquantiles(data, prob=[0.25, 0.75])
        IQR = q[1] - q[0]  # interquartile range
        n = len(data)
        h = 2 * IQR / n ** (1. / 3.)
        return (np.max(data) - np.min(data)) / h

    @staticmethod
    def risk(data):
        import scipy.optimize as optimize

        m, M = np.min(data), np.max(data)

        def f(data):
            def fff(x):  # h is spacing
                h = x[0]
                nbins = (M - m) / h
                binning = np.arange(m, M, h)
                if not len(binning):
                    return float("+inf")
                histo, bincenters = np.histogram(data, binning)
                bincenters = 0.5 * (bincenters[1:] + bincenters[:-1])
                mean = 1. / nbins * np.sum(histo)
                v2 = 1. / nbins * np.sum((histo - mean) ** 2)
                return (2 * mean - v2) / h ** 2
            return fff

        k0 = np.sqrt(len(data))
        h0 = (M - m) / k0
        h = optimize.fmin(f(data), np.array([h0]), disp=False)[0]
        return (M - m) / h

    @staticmethod
    def knuth(data):
        """
        References
        ----------
        .. [1] K. Knuth, "Optimal Data-Based Binning for Histograms", 2006.
               http://arxiv.org/pdf/physics/0605197v1.pdf
        """
        import scipy.optimize as optimize

        def f(data):
            from scipy.special import gammaln

            m, M = np.min(data), np.max(data)
            n = len(data)

            def fff(x):
                k = x[0]  # number of bins
                if k <= 0:
                    return float("+inf")
                binning = np.linspace(m, M, k + 1)
                histo, bincenters = np.histogram(data, binning)

                return -(n * np.log(k) + gammaln(k / 2.) - gammaln(n + k / 2.) +
                         k * gammaln(1. / 2.) + np.sum(gammaln(histo + 0.5)))
            return fff

        k0 = np.sqrt(len(data))
        return optimize.fmin(f(data), np.array([k0]), disp=False)[0]

    @staticmethod
    def wand(data):
        """
        References
        ----------
        .. [1] M. Wand, "Statistical Computing and Graphics", 1997.
               http://web.ipac.caltech.edu/staff/fmasci/home/statistics_refs/OptimumHistogram.pdf
        """
        raise NotImplementedError
