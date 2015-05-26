# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
"""
Taken from example by Zhiyi Liu, zhiyil@fnal.gov
here: http://root.cern.ch/phpBB3/viewtopic.php?f=3&t=6865
and converted into Python
"""
from __future__ import absolute_import

import ROOT

from math import sqrt
from array import array

from .. import Graph
from ...extern.six.moves import range

__all__ = [
    'qqgraph',
]


def qqgraph(h1, h2, quantiles=None):
    """
    Return a Graph of a quantile-quantile (QQ) plot and confidence band
    """
    if quantiles is None:
        quantiles = max(min(len(h1), len(h2)) / 2, 1)
    nq = quantiles
    # position where to compute the quantiles in [0, 1]
    xq = array('d', [0.] * nq)
    # array to contain the quantiles
    yq1 = array('d', [0.] * nq)
    # array to contain the quantiles
    yq2 = array('d', [0.] * nq)

    for i in range(nq):
        xq[i] = float(i + 1) / nq

    h1.GetQuantiles(nq, yq1, xq)
    h2.GetQuantiles(nq, yq2, xq)

    xq_plus = array('d', [0.] * nq)
    xq_minus = array('d', [0.] * nq)
    yq2_plus = array('d', [0.] * nq)
    yq2_minus = array('d', [0.] * nq)

    """
    KS_cv: KS critical value

               1.36
    KS_cv = -----------
             sqrt( N )

    Where 1.36 is for alpha = 0.05 (confidence level 1-5%=95%, about 2 sigma)

    For 1 sigma (alpha=0.32, CL=68%), the value in the nominator is 0.9561,
    it is gotten by GetCriticalValue(1, 1 - 0.68).

    Notes
    -----

    * For 1-sample KS test (data and theoretic), N should be n

    * For 2-sample KS test (2 data set), N should be sqrt(m*n/(m+n))!
      Here is the case m or n (size of samples) should be effective size
      for a histogram

    * Critical value here is valid for only for sample size >= 80 (some
      references say 35) which means, for example, for a unweighted histogram,
      it must have more than 80 (or 35) entries filled and then confidence
      band is reliable.

    """

    esum1 = effective_sample_size(h1)
    esum2 = effective_sample_size(h2)

    # one sigma band
    KS_cv = (critical_value(1, 1 - 0.68) /
             sqrt((esum1 * esum2) / (esum1 + esum2)))

    for i in range(nq):
        # upper limit
        xq_plus[i] = float(xq[i] + KS_cv)
        # lower limit
        xq_minus[i] = float(xq[i] - KS_cv)

    h2.GetQuantiles(nq, yq2_plus, xq_plus)
    h2.GetQuantiles(nq, yq2_minus, xq_minus)

    yq2_err_plus = array('d', [0.] * nq)
    yq2_err_minus = array('d', [0.] * nq)
    for i in range(nq):
        yq2_err_plus[i] = yq2_plus[i] - yq2[i]
        yq2_err_minus[i] = yq2[i] - yq2_minus[i]

    # forget the last point, so number of points: (nq - 1)
    gr = Graph(nq - 1)
    for i in range(nq - 1):
        gr[i] = (yq1[i], yq2[i])
        # confidence level band
        gr.SetPointEYlow(i, yq2_err_minus[i])
        gr.SetPointEYhigh(i, yq2_err_plus[i])

    return gr


def effective_sample_size(h):
    """
    Calculate the effective sample size for a histogram
    the same way as ROOT does.
    """
    sum = 0
    ew = 0
    w = 0
    for bin in h.bins(overflow=False):
        sum += bin.value
        ew = bin.error
        w += ew * ew
    esum = sum * sum / w
    return esum


def critical_value(n, p):
    """
    This function calculates the critical value given
    n and p, and confidence level = 1 - p.
    """
    dn = 1
    delta = 0.5
    res = ROOT.TMath.KolmogorovProb(dn * sqrt(n))
    while res > 1.0001 * p or res < 0.9999 * p:
        if (res > 1.0001 * p):
            dn = dn + delta
        if (res < 0.9999 * p):
            dn = dn - delta
        delta = delta / 2.
        res = ROOT.TMath.KolmogorovProb(dn * sqrt(n))
    return dn
