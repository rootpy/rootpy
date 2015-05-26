# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
from __future__ import absolute_import

import multiprocessing

import ROOT

from . import log; log = log[__name__]
from .. import QROOT, asrootpy
from ..extern.six import string_types
from ..base import NamedObject
from .fit import minimize

__all__ = [
    'Workspace',
]

NCPU = multiprocessing.cpu_count()


class Workspace(NamedObject, QROOT.RooWorkspace):
    _ROOT = QROOT.RooWorkspace

    def __call__(self, *args):
        """
        Need to provide an alternative to RooWorkspace::import since import is
        a reserved word in Python and would be a syntax error.
        """
        return getattr(super(Workspace, self), 'import')(*args)

    def __getitem__(self, name):
        thing = super(Workspace, self).obj(name)
        if thing == None:
            raise ValueError(
                "object named '{0}' does not exist "
                "in the workspace '{1}'".format(name, self.name))
        return asrootpy(thing, warn=False)

    def __contains__(self, name):
        thing = super(Workspace, self).obj(name)
        if thing:
            return True
        return False

    def obj(self, name, cls=None):
        thing = super(Workspace, self).obj(name)
        if thing == None:
            raise ValueError(
                "object named '{0}' does not exist "
                "in the workspace '{1}'".format(name, self.name))
        thing = asrootpy(thing, warn=False)
        if cls is not None and not isinstance(thing, cls):
            raise TypeError(
                "object named '{0}' is not of the correct type: "
                "{1} does not subclass {2}".format(name, thing.__class__, cls))
        return thing

    @property
    def category_functions(self):
        return asrootpy(self.allCatFunctions())

    @property
    def categories(self):
        return asrootpy(self.allCats())

    @property
    def datas(self):
        return self.allData()

    @property
    def functions(self):
        return asrootpy(self.allFunctions())

    @property
    def generic_objects(self):
        return self.allGenericObjects()

    @property
    def pdfs(self):
        return asrootpy(self.allPdfs())

    @property
    def resolution_models(self):
        return asrootpy(self.allResolutionModels())

    @property
    def vars(self):
        return asrootpy(self.allVars())

    def arg(self, name):
        thing = super(Workspace, self).arg(name)
        if thing == None:
            raise ValueError(
                "RooAbsArg named '{0}' does not exist "
                "in the workspace '{1}'".format(name, self.name))
        return asrootpy(thing)

    def argset(self, name):
        thing = super(Workspace, self).argSet(name)
        if thing == None:
            raise ValueError(
                "RooArgSet named '{0}' does not exist "
                "in the workspace '{1}'".format(name, self.name))
        return asrootpy(thing)

    def category(self, name):
        thing = super(Workspace, self).cat(name)
        if thing == None:
            raise ValueError(
                "RooCategory named '{0}' does not exist "
                "in the workspace '{1}'".format(name, self.name))
        return asrootpy(thing)

    def category_function(self, name):
        # Dear RooStats, use camelCase consistently...
        thing = super(Workspace, self).catfunc(name)
        if thing == None:
            raise ValueError(
                "RooAbsCategory named '{0}' does not exist "
                "in the workspace '{1}'".format(name, self.name))
        return thing

    def data(self, name):
        thing = super(Workspace, self).data(name)
        if thing == None:
            raise ValueError(
                "RooAbsData named '{0}' does not exist "
                "in the workspace '{1}'".format(name, self.name))
        return asrootpy(thing)

    def function(self, name):
        thing = super(Workspace, self).function(name)
        if thing == None:
            raise ValueError(
                "RooAbsReal named '{0}' does not exist "
                "in the workspace '{1}'".format(name, self.name))
        return thing

    def pdf(self, name):
        thing = super(Workspace, self).pdf(name)
        if thing == None:
            raise ValueError(
                "RooAbsPdf named '{0}' does not exist "
                "in the workspace '{1}'".format(name, self.name))
        return asrootpy(thing)

    def set(self, name):
        thing = super(Workspace, self).set(name)
        if thing == None:
            raise ValueError(
                "RooArgSet named '{0}' does not exist "
                "in the workspace '{1}'".format(name, self.name))
        return asrootpy(thing)

    def var(self, name):
        thing = super(Workspace, self).var(name)
        if thing == None:
            raise ValueError(
                "RooRealVar named '{0}' does not exist "
                "in the workspace '{1}'".format(name, self.name))
        return asrootpy(thing)

    def fit(self,
            data='obsData',
            model_config='ModelConfig',
            param_const=None,
            param_values=None,
            param_ranges=None,
            poi_const=False,
            poi_value=None,
            poi_range=None,
            extended=False,
            num_cpu=1,
            process_strategy=0,
            offset=False,
            print_level=None,
            return_nll=False,
            **kwargs):
        """
        Fit a pdf to data in a workspace

        Parameters
        ----------

        workspace : RooWorkspace
            The workspace

        data : str or RooAbsData, optional (default='obsData')
            The name of the data or a RooAbsData instance.

        model_config : str or ModelConfig, optional (default='ModelConfig')
            The name of the ModelConfig in the workspace or a
            ModelConfig instance.

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

        extended : bool, optional (default=False)
            If True, add extended likelihood term (False by default)

        num_cpu : int, optional (default=1)
            Parallelize NLL calculation on multiple CPU cores.
            If negative then use all CPU cores.
            By default use only one CPU core.

        process_strategy : int, optional (default=0)
            **Strategy 0:** Divide events into N equal chunks.

            **Strategy 1:** Process event i%N in process N. Recommended for
            binned data with a substantial number of zero-bins, which will be
            distributed across processes more equitably in this strategy.

            **Strategy 2:** Process each component likelihood of a
            RooSimultaneous fully in a single process and distribute components
            over processes. This approach can be benificial if normalization
            calculation time dominates the total computation time of a
            component (since the normalization calculation must be performed
            in each process in strategies 0 and 1. However beware that if the
            RooSimultaneous components do not share many parameters this
            strategy is inefficient: as most minuit-induced likelihood
            calculations involve changing a single parameter, only 1 of the N
            processes will be active most of the time if RooSimultaneous
            components do not share many parameters.

            **Strategy 3:** Follow strategy 0 for all RooSimultaneous
            components, except those with less than 30 dataset entries,
            for which strategy 2 is followed.

        offset : bool, optional (default=False)
            Offset likelihood by initial value (so that starting value of FCN
            in minuit is zero). This can improve numeric stability in
            simultaneously fits with components with large likelihood values.

        print_level : int, optional (default=None)
            The verbosity level for the minimizer algorithm.
            If None (the default) then use the global default print level.
            If negative then all non-fatal messages will be suppressed.

        return_nll : bool, optional (default=False)
            If True then also return the RooAbsReal NLL function that was
            minimized.

        kwargs : dict, optional
            Remaining keyword arguments are passed to the minimize function

        Returns
        -------

        result : RooFitResult
            The fit result.

        func : RooAbsReal
            If return_nll is True, the NLL function is also returned.

        See Also
        --------

        minimize

        """
        if isinstance(model_config, string_types):
            model_config = self.obj(
                model_config, cls=ROOT.RooStats.ModelConfig)
        if isinstance(data, string_types):
            data = self.data(data)
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
                var = self.var(param_name)
                var.setConstant(const)
        if param_values is not None:
            for param_name, param_value in param_values.items():
                var = self.var(param_name)
                var.setVal(param_value)
        if param_ranges is not None:
            for param_name, param_range in param_ranges.items():
                var = self.var(param_name)
                var.setRange(*param_range)

        if print_level < 0:
            msg_service = ROOT.RooMsgService.instance()
            msg_level = msg_service.globalKillBelow()
            msg_service.setGlobalKillBelow(ROOT.RooFit.FATAL)

        args = [
            ROOT.RooFit.Constrain(model_config.GetNuisanceParameters()),
            ROOT.RooFit.GlobalObservables(model_config.GetGlobalObservables())]
        if extended:
            args.append(ROOT.RooFit.Extended(True))
        if offset:
            args.append(ROOT.RooFit.Offset(True))
        if num_cpu != 1:
            if num_cpu == 0:
                raise ValueError("num_cpu must be non-zero")
            if num_cpu < 0:
                num_cpu = NCPU
            args.append(ROOT.RooFit.NumCPU(num_cpu, process_strategy))

        func = pdf.createNLL(data, *args)

        if print_level < 0:
            msg_service.setGlobalKillBelow(msg_level)

        result = minimize(func, print_level=print_level, **kwargs)

        if return_nll:
            return result, func
        return result
