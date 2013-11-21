# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
from nose.plugins.skip import SkipTest
from rootpy.utils.silence import silence_sout

try:
    with silence_sout():
        from ROOT import (RooFit, RooRealVar, RooGaussian, RooArgusBG,
                          RooAddPdf, RooArgList, RooArgSet)
    from rootpy.stats import mute_roostats; mute_roostats()
    from rootpy.stats import Workspace
except ImportError:
    raise SkipTest("ROOT is not compiled with RooFit and RooStats enabled")

from rootpy.io import TemporaryFile
from nose.tools import assert_false


def test_correlated_values():

    try:
        import uncertainties
    except ImportError:
        raise SkipTest("uncertainties package is not installed")
    from rootpy.stats.correlated_values import correlated_values

    # construct pdf and toy data following example at
    # http://root.cern.ch/drupal/content/roofit

    # --- Observable ---
    mes = RooRealVar("mes", "m_{ES} (GeV)", 5.20, 5.30)

    # --- Parameters ---
    sigmean = RooRealVar("sigmean", "B^{#pm} mass", 5.28, 5.20, 5.30)
    sigwidth = RooRealVar("sigwidth", "B^{#pm} width", 0.0027, 0.001, 1.)

    # --- Build Gaussian PDF ---
    signal = RooGaussian("signal", "signal PDF", mes, sigmean, sigwidth)

    # --- Build Argus background PDF ---
    argpar = RooRealVar("argpar", "argus shape parameter", -20.0, -100., -1.)
    background = RooArgusBG("background", "Argus PDF",
                            mes, RooFit.RooConst(5.291), argpar)

    # --- Construct signal+background PDF ---
    nsig = RooRealVar("nsig", "#signal events", 200, 0., 10000)
    nbkg = RooRealVar("nbkg", "#background events", 800, 0., 10000)
    model = RooAddPdf("model", "g+a",
                      RooArgList(signal,background),
                      RooArgList(nsig,nbkg))

    # --- Generate a toyMC sample from composite PDF ---
    data = model.generate(RooArgSet(mes), 2000)

    # --- Perform extended ML fit of composite PDF to toy data ---
    fitresult = model.fitTo(data, RooFit.Save(), RooFit.PrintLevel(-1))

    nsig, nbkg = correlated_values(["nsig", "nbkg"], fitresult)

    # Arbitrary math expression according to what the `uncertainties`
    # package supports, automatically computes correct error propagation
    sum_value = nsig + nbkg
    value, error = sum_value.nominal_value, sum_value.std_dev

    workspace = Workspace(name='workspace')
    # import the data
    assert_false(workspace(data))
    with TemporaryFile():
        workspace.Write()
