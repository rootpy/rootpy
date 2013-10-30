# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
from __future__ import absolute_import

import ROOT

from . import log; log = log[__name__]
from ..base import Object
from .. import QROOT, asrootpy

__all__ = [
    'ArgSet',
]


class _CollectionBase(Object):

    def __getitem__(self, name):
        thing = self.find(name)
        if thing == None:
            raise ValueError(
                "argument '{0}' is not contained "
                "in the RooArgSet '{1}'".format(name, self.GetName()))
        return asrootpy(thing, warn=False)

    def __contains__(self, value):
        if isinstance(value, basestring):
            try:
                thing = self[value]
            except ValueError:
                return False
            return True
        # RooAbsArg
        return self.contains(value)

    def __iter__(self):
        start = self.fwdIterator()
        for i in xrange(len(self)):
            yield asrootpy(start.next(), warn=False)

    def __len__(self):
        return self.getSize()

    def __eq__(self, other):
        return self.equals(other)

    def find(self, name):
        thing = super(_CollectionBase, self).find(name)
        if thing == None:
            return None
        return asrootpy(thing, warn=False)

    def first(self):
        thing = super(_CollectionBase, self).first()
        if thing == None:
            return None
        return asrootpy(thing, warn=False)


class ArgSet(_CollectionBase, QROOT.RooArgSet):

    _ROOT = QROOT.RooArgSet
