# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
from __future__ import absolute_import

import ROOT

from . import log; log = log[__name__]

__all__ = [
    'fit_workspace',
    'minimize',
]


def fit_workspace(workspace,
                  data_name='obsData',
                  model_config_name='ModelConfig',
                  param_const=None,
                  param_values=None,
                  param_ranges=None,
                  poi_const=False,
                  poi_value=None,
                  poi_range=None):
    """
    Fit a pdf to data in a workspace

    Parameters
    ----------

    workspace : RooWorkspace
        The workspace

    data_name : str, optional (default='obsData')
        The name of the data

    model_config_name : str, optional (default='ModelConfig')
        The name of the ModelConfig in the workspace

    param_const : dict, optional (default=None)
        A dict mapping parameter names to booleans setting
        the const state of the parameter

    param_values : dict, optional (default=None)
        A dict mapping parameter names to values

    param_ranges : dict, optional (default=None)
        A dict mapping parameter names to 2-tuples defining the ranges

    poi_const : bool, optional (default=False)
        If True, then make the parameter of interest (POI) constant

    poi_value : float, optional (default=None)
        If not None, then set the POI to this value

    poi_range : tuple, optional (default=None)
        If not None, then set the range of the POI with this 2-tuple

    Returns
    -------

    result : RooFitResult
        The fit result

    See Also
    --------

    minimize

    """
    model_config = workspace.obj(model_config_name)
    data = workspace.data(data_name)

    pdf = model_config.GetPdf()

    poi = model_config.GetParametersOfInterest().first()
    poi.setConstant(poi_const)
    if poi_value is not None:
        poi.setVal(poi_value)
    if poi_range is not None:
        poi.setRange(*poi_range)

    if param_const is not None:
        for param_name, const in param_const.items():
            var = workspace.var(param_name)
            var.setConstant(const)
    if param_values is not None:
        for param_name, param_value in param_values.items():
            var = workspace.var(param_name)
            var.setVal(param_value)
    if param_ranges is not None:
        for param_name, param_range in param_ranges.items():
            var = workspace.var(param_name)
            var.setRange(*param_range)

    func = pdf.createNLL(data)
    result = minimize(func)
    return result


def minimize(func):
    """
    Minimize a RooAbsReal function

    Parameters
    ----------

    func : RooAbsReal
        The function to minimize

    Returns
    -------

    result : RooFitResult
        The fit result

    """
    llog = log['minimize']

    print_level = ROOT.Math.MinimizerOptions.DefaultPrintLevel()
    msg_service = ROOT.RooMsgService.instance()
    msg_level = msg_service.globalKillBelow()
    if print_level < 0:
        msg_service.setGlobalKillBelow(ROOT.RooFit.FATAL)

    strat = ROOT.Math.MinimizerOptions.DefaultStrategy()
    minim = ROOT.RooMinimizer(func)
    minim.setStrategy(strat)
    minim.setPrintLevel(print_level)

    status = minim.minimize(
        ROOT.Math.MinimizerOptions.DefaultMinimizerType(),
        ROOT.Math.MinimizerOptions.DefaultMinimizerAlgo())

    for itry in xrange(2):
        if status not in (0, 1) and strat < 2:
            strat += 1
            log.warning(
                "Fit failed with status {0:d}. "
                "Retrying with strategy {1:d}".format(status, strat))
            minim.setStrategy(strat)
            status = minim.minimize(
                ROOT.Math.MinimizerOptions.DefaultMinimizerType(),
                ROOT.Math.MinimizerOptions.DefaultMinimizerAlgo())

    if status not in (0, 1):
        log.warning("Fit failed with status {0:d}".format(status))
        min_type = ROOT.Math.MinimizerOptions.DefaultMinimizerType()
        new_min_type = 'Minuit' if min_type == 'Minuit2' else 'Minuit2'
        log.info("Switching minuit type from {0} to {1}".format(
            curr_min_type, new_min_type))

        ROOT.Math.MinimizerOptions.SetDefaultMinimizer(new_min_type)
        strat = 1 # ROOT.Math.MinimizerOptions.DefaultStrategy()
        minim.setStrategy(strat)

        status = minim.minimize(
            ROOT.Math.MinimizerOptions.DefaultMinimizerType(),
            ROOT.Math.MinimizerOptions.DefaultMinimizerAlgo())

        for itry in xrange(2):
            if status not in (0, 1) and strat < 2:
                strat += 1
                log.warning(
                    "Fit failed with status {0:d}. "
                    "Retrying with strategy {1:d}".format(status, strat))
                minim.setStrategy(strat)
                status = minim.minimize(
                    ROOT.Math.MinimizerOptions.DefaultMinimizerType(),
                    ROOT.Math.MinimizerOptions.DefaultMinimizerAlgo())

        ROOT.Math.MinimizerOptions.SetDefaultMinimizer(min_type)

    if status == 0:
        llog.info("successful fit")
    else:
        llog.warning("fit failed with status {0:d}".format(status))

    if print_level < 0:
        msg_service.setGlobalKillBelow(msg_level)

    return minim.save()
