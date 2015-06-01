# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
from __future__ import absolute_import

import ROOT

from . import log; log = log[__name__]
from ..extern.six.moves import range
from ..extern.six import string_types
from ..base import Object
from .. import QROOT, asrootpy

__all__ = [
    'ArgSet',
    'ArgList',
]


class _CollectionBase(object):

    def __getitem__(self, name):
        thing = self.find(name)
        if thing == None:
            raise ValueError(
                "argument '{0}' is not contained "
                "in the RooArgSet '{1}'".format(name, self.GetName()))
        return asrootpy(thing, warn=False)

    def __contains__(self, value):
        if isinstance(value, string_types):
            try:
                thing = self[value]
            except ValueError:
                return False
            return True
        # RooAbsArg
        return self.contains(value)

    def __iter__(self):
        start = self.fwdIterator()
        for i in range(len(self)):
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

    def __repr__(self):
        return "{0}(name='{1}', {2})".format(
            self.__class__.__name__,
            self.GetName(),
            repr(list(self)))

    @property
    def name(self):
        return self.GetName()

    @name.setter
    def name(self, value):
        # ROOT, why is your API so inconsistent?
        # We have GetName() and setName() here...
        self.setName(value)


class ArgSet(_CollectionBase, Object, QROOT.RooArgSet):
    _ROOT = QROOT.RooArgSet


class ArgList(_CollectionBase, Object, QROOT.RooArgList):
    _ROOT = QROOT.RooArgList
