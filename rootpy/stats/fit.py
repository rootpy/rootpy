# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
from __future__ import absolute_import

import ROOT

from . import log; log = log[__name__]
from .. import QROOT, asrootpy

__all__ = [
    'minimize',
    'Minimizer',
    'FitResult',
]


def minimize(func,
             minimizer_type=None,
             minimizer_algo=None,
             strategy=None,
             retry=0,
             scan=False,
             print_level=None):
    """
    Minimize a RooAbsReal function

    Parameters
    ----------

    func : RooAbsReal
        The function to minimize

    minimizer_type : string, optional (default=None)
        The minimizer type: "Minuit" or "Minuit2".
        If None (the default) then use the current global default value.

    minimizer_algo : string, optional (default=None)
        The minimizer algorithm: "Migrad", etc.
        If None (the default) then use the current global default value.

    strategy : int, optional (default=None)
        Set the MINUIT strategy. Accepted values
        are 0, 1, and 2 and represent MINUIT strategies for dealing
        most efficiently with fast FCNs (0), expensive FCNs (2)
        and 'intermediate' FCNs (1). If None (the default) then use
        the current global default value.

    retry : int, optional (default=0)
        Number of times to retry failed minimizations. The strategy is
        incremented to a maximum of 2 from its initial value and remains at 2
        for additional retries.

    scan : bool, optional (default=False)
        If True then run Minuit2's scan algorithm before running the main
        ``minimizer_algo`` ("Migrad").

    print_level : int, optional (default=None)
        The verbosity level for the minimizer algorithm.
        If None (the default) then use the global default print level.
        If negative then all non-fatal messages will be suppressed.

    Returns
    -------

    minimizer : RooMinimizer
        The minimizer. Get the RooFitResult with ``minimizer.save()``.

    """
    llog = log['minimize']

    min_opts = ROOT.Math.MinimizerOptions
    if minimizer_type is None:
        minimizer_type = min_opts.DefaultMinimizerType()
    if minimizer_algo is None:
        minimizer_algo = min_opts.DefaultMinimizerAlgo()
    if strategy is None:
        strategy = min_opts.DefaultStrategy()
    if print_level is None:
        print_level = min_opts.DefaultPrintLevel()

    if print_level < 0:
        msg_service = ROOT.RooMsgService.instance()
        msg_level = msg_service.globalKillBelow()
        msg_service.setGlobalKillBelow(ROOT.RooFit.FATAL)

    minim = Minimizer(func)
    minim.setPrintLevel(print_level)
    minim.setStrategy(strategy)

    if scan:
        llog.info("running scan algorithm ...")
        minim.minimize('Minuit2', 'Scan')

    llog.info("minimizing with {0} {1} using strategy {2}".format(
        minimizer_type, minimizer_algo, strategy))
    status = minim.minimize(minimizer_type, minimizer_algo)

    iretry = 0
    while iretry < retry and status not in (0, 1):
        if strategy < 2:
            strategy += 1
            minim.setStrategy(strategy)
        llog.warning("minimization failed with status {0:d}".format(status))
        llog.info("retrying minimization with strategy {0:d}".format(strategy))
        status = minim.minimize(minimizer_type, minimizer_algo)

    if status in (0, 1):
        llog.info("found minimum")
    else:
        llog.warning("minimization failed with status {0:d}".format(status))

    if print_level < 0:
        msg_service.setGlobalKillBelow(msg_level)

    return minim


class Minimizer(QROOT.RooMinimizer):
    _ROOT = QROOT.RooMinimizer

    def save(self, *args, **kwargs):
        return asrootpy(super(Minimizer, self).save(*args, **kwargs))


class FitResult(QROOT.RooFitResult):
    _ROOT = QROOT.RooFitResult

    @property
    def constant_params(self):
        return asrootpy(super(FitResult, self).constPars())

    @property
    def final_params(self):
        return asrootpy(super(FitResult, self).floatParsFinal())

    @property
    def initial_params(self):
        return asrootpy(super(FitResult, self).floatParsInit())

    @property
    def covariance_matrix(self):
        return asrootpy(super(FitResult, self).covarianceMatrix())

    @property
    def correlation_matrix(self):
        return asrootpy(super(FitResult, self).correlationMatrix())

    def reduced_covariance_matrix(self, params):
        return asrootpy(
            super(FitResult, self).reducedCovarianceMatrix(params))

    def conditional_covariance_matrix(self, params):
        return asrootpy(
            super(FitResult, self).conditionalCovarianceMatrix(params))
