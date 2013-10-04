# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
from __future__ import absolute_import

import uncertainties as U

from .. import asrootpy

__all__ = [
    'as_ufloat',
    'correlated_values',
]


def as_ufloat(roorealvar):
    """
    Cast a `RooRealVar` to an `uncertainties.ufloat`
    """
    if isinstance(roorealvar, (U.AffineScalarFunc, U.Variable)):
        return roorealvar
    return U.ufloat((roorealvar.getVal(), roorealvar.getError()))


def correlated_values(param_names, roofitresult):
    """
    Return symbolic values from a `RooFitResult` taking into account covariance

    This is useful for numerically computing the uncertainties for expressions
    using correlated values arising from a fit.

    Parameters
    ----------

    param_names: list of strings
        A list of parameters to extract from the result. The order of the names
        is the order of the return value.

    roofitresult : RooFitResult
        A RooFitResult from a fit.

    Returns
    -------

    list of correlated values from the uncertainties package.

    Examples
    --------

    .. sourcecode:: python

        # Fit a pdf to a histogram
        pdf = some_roofit_pdf_with_variables("f(x, a, b, c)")
        fitresult = pdf.fitTo(histogram, ROOT.RooFit.Save())
        a, b, c = correlated_values(["a", "b", "c"], fitresult)
        # Arbitrary math expression according to what the `uncertainties`
        # package supports, automatically computes correct error propagation
        sum_value = a + b + c
        value, error = sum_value.nominal_value, sum_value.std_dev()

    """
    pars = roofitresult.floatParsFinal()
    #pars.Print()
    pars = [pars[i] for i in range(pars.getSize())]
    parnames = [p.GetName() for p in pars]

    values = [(p.getVal(), p.getError()) for p in pars]
    #values = [as_ufloat(p) for p in pars]
    matrix = asrootpy(roofitresult.correlationMatrix()).to_numpy()

    uvalues = U.correlated_values_norm(values, matrix.tolist())
    uvalues = dict((n, v) for n, v in zip(parnames, uvalues))

    assert all(n in uvalues for n in parnames), (
        "name {0} isn't in parameter list {1}".format(n, parnames))

    # Return a tuple in the order it was asked for
    return tuple(uvalues[n] for n in param_names)
