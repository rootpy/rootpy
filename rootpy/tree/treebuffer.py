# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
from __future__ import absolute_import

import sys
import re

import ROOT

from collections import Hashable
try:
    from collections import OrderedDict
except ImportError: # py 2.6
    from ..extern.ordereddict import OrderedDict

from . import log
from .. import lookup_by_name, create, stl
from ..base import Object
from .treetypes import Scalar, Array, Int, Char, UChar, BaseCharArray
from .treeobject import TreeCollection, TreeObject, mix_classes


__all__ = [
    'TreeBuffer',
]


class TreeBuffer(OrderedDict):
    """
    A dictionary mapping branch names to values
    """
    ARRAY_PATTERN = re.compile('^(?P<type>[^\[]+)\[(?P<length>\d+)\]$')

    def __init__(self,
                 branches=None,
                 tree=None,
                 ignore_unsupported=False):
        super(TreeBuffer, self).__init__()
        self._fixed_names = {}
        self._branch_cache = {}
        self._branch_cache_event = {}
        self._tree = tree
        self._ignore_unsupported = ignore_unsupported
        self._current_entry = 0
        self._collections = {}
        self._objects = []
        self._entry = Int(0)
        if branches is not None:
            self.__process(branches)
        self._inited = True

    @classmethod
    def __clean(cls, branchname):
        # Replace invalid characters with '_'
        branchname = re.sub('[^0-9a-zA-Z_]', '_', branchname)
        # Remove leading characters until we find a letter or underscore
        return re.sub('^[^a-zA-Z_]+', '', branchname)

    def __process(self, branches):
        if not branches:
            return
        if not isinstance(branches, dict):
            try:
                branches = dict(branches)
            except TypeError:
                raise TypeError(
                    "branches must be a dict or anything "
                    "the dict constructor accepts")
        processed = []
        for name, vtype in branches.items():
            if name in processed:
                raise ValueError(
                    "duplicate branch name `{0}`".format(name))
            processed.append(name)
            obj = None
            array_match = re.match(TreeBuffer.ARRAY_PATTERN, vtype)
            if array_match:
                vtype = array_match.group('type') + '[]'
                length = int(array_match.group('length'))
                # try to lookup type in registry
                cls = lookup_by_name(vtype)
                if cls is not None:
                    # special case for [U]Char and [U]CharArray with
                    # null-termination
                    if issubclass(cls, BaseCharArray):
                        if length == 2:
                            obj = cls.scalar()
                        elif length == 1:
                            raise ValueError(
                                "char branch `{0}` is not "
                                "null-terminated".format(name))
                        else:
                            # leave slot for null-termination
                            obj = cls(length)
                    else:
                        obj = cls(length)
            else:
                # try to lookup type in registry
                cls = lookup_by_name(vtype)
                if cls is not None:
                    obj = cls()
                else:
                    # try to create ROOT.'vtype'
                    obj = create(vtype)
                    if obj is None:
                        # try to generate this type
                        cpptype = stl.CPPType.try_parse(vtype)
                        if cpptype and cpptype.is_template:
                            obj = cpptype.cls()
            if obj is None:
                if not self._ignore_unsupported:
                    raise TypeError(
                        "branch `{0}` has unsupported "
                        "type `{1}`".format(name, vtype))
                else:
                    log.warning(
                        "ignoring branch `{0}` with "
                        "unsupported type `{1}`".format(name, vtype))
            else:
                self[name] = obj

    def reset(self):
        if sys.version_info[0] < 3:
            iter_values = self.itervalues()
        else:
            iter_values = self.values()
        for value in iter_values:
            if isinstance(value, (Scalar, Array)):
                value.reset()
            elif isinstance(value, Object):
                value._ROOT.__init__(value)
            elif isinstance(value, (ROOT.TObject, ROOT.ObjectProxy)):
                value.__init__()
            else:
                # there should be no other types of objects in the buffer
                raise TypeError(
                    "cannot reset object of type `{0}`".format(type(value)))

    def update(self, branches=None):
        if branches is None:
            # don't break super update
            return
        if isinstance(branches, TreeBuffer):
            self._entry = branches._entry
            for name, value in branches.items():
                super(TreeBuffer, self).__setitem__(name, value)
            self._fixed_names.update(branches._fixed_names)
        else:
            self.__process(branches)

    def set_tree(self, tree=None):
        self._branch_cache = {}
        self._branch_cache_event = {}
        self._tree = tree
        self._current_entry = 0

    def next_entry(self):
        super(TreeBuffer, self).__setattr__('_branch_cache_event', {})
        self._current_entry += 1

    def get_with_read_if_cached(self, attr):
        if self._tree is not None:
            try:
                branch = self._branch_cache[attr]
            except KeyError:
                # branch is being accessed for the first time
                branch = self._tree.GetBranch(attr)
                if not branch:
                    raise AttributeError
                self._branch_cache[attr] = branch
                self._tree.AddBranchToCache(branch)
            if sys.version_info[0] >= 3 and not isinstance(branch, Hashable):
                # PyROOT missing __hash__ for Python 3
                branch.__class__.__hash__ = object.__hash__
            if branch not in self._branch_cache_event:
                # branch is being accessed for the first time in this entry
                branch.GetEntry(self._current_entry)
                self._branch_cache_event[branch] = None
        return super(TreeBuffer, self).__getitem__(attr)

    def __setitem__(self, name, value):
        # for a key to be used as an attr it must be a valid Python identifier
        fixed_name = TreeBuffer.__clean(name)
        if fixed_name in dir(self) or fixed_name.startswith('_'):
            raise ValueError("illegal branch name: `{0}`".format(name))
        if fixed_name != name:
            self._fixed_names[fixed_name] = name
        super(TreeBuffer, self).__setitem__(name, value)

    def __getitem__(self, name):
        return self.get_with_read_if_cached(name)

    def __setattr__(self, attr, value):
        """
        Maps attributes to values.
        Only if we are initialized
        """
        # this test allows attributes to be set in the __init__ method
        # any normal attributes are handled normally
        if '_inited' not in self.__dict__ or attr in self.__dict__:
            return super(TreeBuffer, self).__setattr__(attr, value)
        elif attr in self:
            variable = self.get_with_read_if_cached(attr)
            if isinstance(variable, (Scalar, Array)):
                variable.set(value)
                return
            elif isinstance(variable, Object):
                variable.copy_from(value)
                return
            elif isinstance(variable, (ROOT.TObject, ROOT.ObjectProxy)):
                # copy constructor
                variable.__init__(value)
                return
            raise TypeError(
                "cannot set attribute `{0}` of `{1}` instance".format(
                    attr, self.__class__.__name__))
        raise AttributeError(
            "`{0}` instance has no attribute `{1}`".format(
                self.__class__.__name__, attr))

    def __getattr__(self, attr):
        if '_inited' not in self.__dict__:
            raise AttributeError(
                "`{0}` instance has no attribute `{1}`".format(
                    self.__class__.__name__, attr))
        if attr in self._fixed_names:
            attr = self._fixed_names[attr]
        try:
            variable = self.get_with_read_if_cached(attr)
            if isinstance(variable, Scalar):
                return variable.value
            return variable
        except (KeyError, AttributeError):
            raise AttributeError(
                "{0} instance has no attribute `{1}`".format(
                    self.__class__.__name__, attr))

    def reset_collections(self):
        if sys.version_info[0] < 3:
            for coll in self._collections.iterkeys():
                coll.reset()
        else:
            for coll in self._collections.keys():
                coll.reset()

    def define_collection(self, name, prefix, size, mix=None):
        coll = TreeCollection(self, name, prefix, size, mix=mix)
        object.__setattr__(self, name, coll)
        self._collections[coll] = (name, prefix, size, mix)
        return coll

    def define_object(self, name, prefix, mix=None):
        cls = TreeObject
        if mix is not None:
            cls = mix_classes(TreeObject, mix)
        obj = cls(self, name, prefix)
        object.__setattr__(self, name, obj)
        self._objects.append((name, prefix, mix))
        return obj

    def set_objects(self, other):
        for args in other._objects:
            self.define_object(*args)
        if sys.version_info[0] < 3:
            for args in other._collections.itervalues():
                self.define_collection(*args)
        else:
            for args in other._collections.values():
                self.define_collection(*args)

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        rep = ''
        for name, value in self.items():
            rep += '{0} -> {1}\n'.format(name, repr(value))
        return rep
