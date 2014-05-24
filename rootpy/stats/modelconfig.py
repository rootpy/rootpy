# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
from __future__ import absolute_import

import ROOT

from . import log; log = log[__name__]
from .. import QROOT, asrootpy
from ..base import NamedObject

__all__ = [
    'ModelConfig',
]


class ModelConfig(NamedObject, QROOT.RooStats.ModelConfig):
    _ROOT = QROOT.RooStats.ModelConfig

    def GetPdf(self):
        return asrootpy(super(ModelConfig, self).GetPdf())

    @property
    def workspace(self):
        return asrootpy(self.GetWorkspace())

    @workspace.setter
    def workspace(self, value):
        self.SetWorkspace(value)

    @property
    def pdf(self):
        return self.GetPdf()

    @pdf.setter
    def pdf(self, value):
        self.SetPdf(value)

    @property
    def prior_pdf(self):
        return self.GetPriorPdf()

    @prior_pdf.setter
    def prior_pdf(self, value):
        self.SetPriorPdf(value)

    @property
    def proto_data(self):
        return self.GetProtoData()

    @proto_data.setter
    def proto_data(self, value):
        self.SetProtoData(value)

    @property
    def snapshot(self):
        return self.GetSnapshot()

    @snapshot.setter
    def snapshot(self, value):
        self.SetSnapshot(value)

    @property
    def conditional_observables(self):
        return asrootpy(self.GetConditionalObservables())

    @conditional_observables.setter
    def conditional_observables(self, value):
        self.SetConditionalObservables(value)

    @property
    def constraint_parameters(self):
        return asrootpy(self.GetConstraintParameters())

    @constraint_parameters.setter
    def constraint_parameters(self, value):
        self.SetConstraintParameters(value)

    @property
    def global_observables(self):
        return asrootpy(self.GetGlobalObservables())

    @global_observables.setter
    def global_observables(self, value):
        self.SetGlobalObservables(value)

    @property
    def nuisance_parameters(self):
        return asrootpy(self.GetNuisanceParameters())

    @nuisance_parameters.setter
    def nuisance_parameters(self, value):
        self.SetNuisanceParameters(value)

    @property
    def observables(self):
        return asrootpy(self.GetObservables())

    @observables.setter
    def observables(self, value):
        self.SetObservables(value)

    @property
    def poi(self):
        return asrootpy(self.GetParametersOfInterest())

    @poi.setter
    def poi(self, value):
        self.SetParametersOfInterest(value)
