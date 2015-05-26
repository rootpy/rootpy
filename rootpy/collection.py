# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
from __future__ import absolute_import

from collections import namedtuple

import ROOT

from . import QROOT, asrootpy
from .extern.six.moves import range
from .base import Object

__all__ = [
    'List',
    'ObjArray',
]


TListItemWithOption = namedtuple("TListItemWithOption", "item option")


class List(Object, QROOT.TList):
    """
    rootpy wrapper on ROOT's TList. Primarily provides the ability to do slice
    assignments whilst preserving options, which makes it useful for
    manipulating TLists such as ROOT's ``TCanvas::GetListOfPrimitives``.

    Note: this class is rather inefficient as is only intended for manipulating
    small numbers of objects. In modern computing, a linked list wouldn't be
    used in this case. Since a TList is what we have, this provides some sane
    ways to use them.
    """
    _ROOT = QROOT.TList

    @property
    def as_list_with_options(self):
        """
        Similar to list(self) except elements which have an option associated
        with them are returned as a ``TListItemWithOption``
        """
        it = ROOT.TIter(self)
        elem = it.Next()
        result = []
        while elem:
            if it.GetOption():
                result.append(TListItemWithOption(elem, it.GetOption()))
            else:
                result.append(elem)
            elem = it.Next()
        return result

    def Add(self, value, *optional):
        """
        Overload ROOT's basic TList::Add to support supplying
        TListItemWithOption
        """
        if isinstance(value, TListItemWithOption):
            if optional:
                raise RuntimeError(
                    "option specified along with "
                    "TListItemWithOption. Specify one or the "
                    "other but not both.")
            return super(List, self).Add(value.item, value.option)
        return super(List, self).Add(value, *optional)

    def __setitem__(self, idx, desired):
        """
        Support slice assignment to a TList
        """
        if not isinstance(idx, slice):
            super(List, self)[idx] = desired

        if not isinstance(desired, (list, tuple)):
            raise NotImplementedError(
                "Only support list or tuple in slice assignment")

        # Implementation: completely clear the list and rebuild it.
        # If we own objects, manually delete the ones which don't get re-added
        # to the list.

        original_values = self.as_list_with_options

        self.Clear("nodelete")

        first_idx, last_idx, stride = idx.indices(len(original_values))

        newlist = (original_values[:first_idx:stride]
                   + list(desired)
                   + original_values[last_idx::stride])

        for item in newlist:
            # TODO: Potentially fix up the "same" keyword intelligently *if*
            #       the first item is a TFrame, we know we're probably a list
            #       of items which is being drawn. For example, we might want
            #       objects which are specified repeatedly to use the "same"
            #       keyword the second time, or to ensure that the first
            #       does not have the "same" keyword.
            pass

        # Set of objects which were used (and don't need deleting)
        added = set()

        # Rebuild the list
        for value in newlist:
            self.Add(value)
            added.add(value)

        to_disown = set(original_values) - set(added)
        if self.IsOwner() and to_disown:
            # These items need deleting if we own them.
            # Add them to a temporary TList which we then delete in order to
            # get the correct deletion semantics.

            templist = List()
            templist.SetOwner()
            for item in to_disown:
                templist.Add(item)

            # Causes deletion of heap based objects with the usual root
            # semantics.
            templist.Clear()

    def __getitem__(self, idx):
        """
        Similar to list(self)[idx] except it uses
        ``List.as_list_with_options``.
        """
        return self.as_list_with_options[idx]

    def __iter__(self):
        for item in super(List, self).__iter__():
            yield asrootpy(item)

    def __repr__(self):
        return "rootpy.List{0}".format(list(self))


class ObjArray(Object, QROOT.TObjArray):
    """
    Make ObjArray return asrootpy'd versions of the objects contained within.
    """
    _ROOT = QROOT.TObjArray

    # TODO: override other TObjArray methods which return TObject*
    def At(self, idx):
        return asrootpy(super(ObjArray, self).At(idx))

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return [self[i] for i in range(*idx.indices(len(self)))]
        return self.At(idx)

    def __iter__(self):
        for item in super(ObjArray, self).__iter__():
            yield asrootpy(item)
