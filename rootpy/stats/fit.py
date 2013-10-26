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
                  poi_range=None,
                  **kwargs):
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

    kwargs : dict, optional
        Remaining keyword arguments are passed to the minimize function

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

    pois = model_config.GetParametersOfInterest()
    if pois.getSize() > 0:
        poi = pois.first()
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
    return minimize(func, **kwargs)


def minimize(func,
             minimizer_type=None,
             minimizer_algo=None,
             strategy=None,
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

    print_level : int, optional (default=None)
        The verbosity level for the minimizer algorithm.
        If None (the default) then use the global default print level.

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

    msg_service = ROOT.RooMsgService.instance()
    msg_level = msg_service.globalKillBelow()
    if print_level < 0:
        msg_service.setGlobalKillBelow(ROOT.RooFit.FATAL)

    minim = ROOT.RooMinimizer(func)
    minim.setStrategy(strategy)
    minim.setPrintLevel(print_level)

    status = minim.minimize(minimizer_type, minimizer_algo)

    if status == 0:
        llog.info("successful fit")
    else:
        llog.warning("fit failed with status {0:d}".format(status))

    if print_level < 0:
        msg_service.setGlobalKillBelow(msg_level)

    return minim
