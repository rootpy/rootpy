import ROOT
from ROOT import RooMinimizer

from . import log; log = log[__name__]


def fit(pdf, obs_data):

    # Construct the negative log likelihood of the PDF
    nll = pdf.createNLL(obs_data)

    # Create and configure the minimizer
    minim = RooMinimizer(nll)
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
        log.info("fitting iteration %d" % tries)
        status = minim.minimize(minimizer, algorithm)
        log.info("fitting status = %d" % status)
        if status % 1000 == 0:
            # ignore errors from Improve
            break
        elif tries == 1:
                log.info("performing an initial scan")
                minim.minimize(minimizer, "Scan")
        elif tries == 2:
            if ROOT.Math.MinimizerOptions.DefaultStrategy() == 0:
                log.info("trying strategy 1")
                minim.setStrategy(1)
            else:
                # skip this trial if strategy is already 1
                tries += 1
        elif tries == 3:
            log.info("trying improved MIGRAD")
            minimizer = "Minuit";
            algorithm = "migradimproved";

    fit_result = minim.save()
    return fit_result
