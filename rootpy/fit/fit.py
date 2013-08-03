# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
import ROOT
from . import log; log = log[__name__]

__all__ = [
    'nll_fit',
    'minimize',
]


def nll_fit(pdf, obs_data):

    # Minimize the negative log likelihood of the PDF
    return minimize(pdf.createNLL(obs_data))


def minimize(func):

    llog = log['minimize']
    # Create and configure the minimizer
    minim = ROOT.RooMinimizer(func)
    strategy = ROOT.Math.MinimizerOptions.DefaultStrategy()
    minim.setStrategy(strategy)
    tol = ROOT.Math.MinimizerOptions.DefaultTolerance()
    minim.setEps(max(tol, 1.))
    minim.setPrintLevel(0)
    minim.optimizeConst(2)
    minimizer = ROOT.Math.MinimizerOptions.DefaultMinimizerType()
    algorithm = ROOT.Math.MinimizerOptions.DefaultMinimizerAlgo()
    status = -1

    tries = 1
    maxtries = 4

    # Perform the minimization
    while tries <= maxtries:
        llog.info("fitting iteration {0:d}".format(tries))
        status = minim.minimize(minimizer, algorithm)
        llog.info("fitting status = {0:d}".format(status))
        if status % 1000 == 0:
            # ignore errors from Improve
            break
        elif tries == 1:
            llog.info("performing an initial scan")
            minim.minimize(minimizer, "Scan")
        elif tries == 2:
            if ROOT.Math.MinimizerOptions.DefaultStrategy() == 0:
                llog.info("trying strategy 1")
                minim.setStrategy(1)
            else:
                # skip this trial if strategy is already 1
                tries += 1
        elif tries == 3:
            llog.info("trying improved MIGRAD")
            minimizer = "Minuit"
            algorithm = "migradimproved"

    return minim.save()
