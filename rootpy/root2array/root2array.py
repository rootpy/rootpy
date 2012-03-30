"""
This module includes:
* conversion of TTrees into NumPy arrays. Done
* conversion of TTrees into carrays (http://pypi.python.org/pypi/carray). TODO
"""

from ..types import Variable, convert
from ..utils import asrootpy
import numpy as np
from .root_numpy import pyroot2rec, pyroot2array


def tree_to_ndarray_c(trees, branches=None):

    if not isinstance(trees, (list, tuple)):
        trees = [trees]
    return np.concatenate([pyroot2array(tree, branches) for tree in trees])


def tree_to_recarray_c(trees, branches=None):

    if not isinstance(trees, (list, tuple)):
        trees = [trees]
    return np.concatenate([pyroot2rec(tree, branches) for tree in trees])


def recarray_to_ndarray(recarray, dtype=np.float32):
    """
    Convert a numpy.recarray into a numpy.ndarray
    """
    ndarray = np.empty((len(recarray), len(recarray.dtype)), dtype=dtype)
    for idx, field in enumerate(recarray.dtype.names):
        ndarray[:,idx] = recarray[field]
    return ndarray


def tree_to_recarray(trees, branches=None,
                     use_cache=False, cache_size=1000000,
                     include_weight=False,
                     weight_name='weight',
                     weight_dtype='f4'):
    """
    Convert a tree or a list of trees into a numpy.recarray
    with fields corresponding to the tree branches
    """
    if not isinstance(trees, (list, tuple)):
        trees = [trees]
    trees = [asrootpy(tree) for tree in trees]
    # if branches is None then select only branches with basic types
    # i.e. no vectors or other special objects
    tree = trees[0]
    _branches = {}
    if branches is None:
        branches = []
        for name, value in tree.buffer.items():
            if isinstance(value, Variable):
                _branches[name] = value
                branches.append(name)
    else:
        if len(set(branches)) != len(branches):
            raise ValueError("branches contains duplicates")
        for branch in branches:
            if branch not in tree.buffer:
                raise ValueError("Branch %s does not exist in tree" % branch)
            value = tree.buffer[branch]
            if not isinstance(value, Variable):
                raise TypeError("Branch %s is not a basic type: %s" %
                                (branch, type(value)))
            _branches[branch] = value
    if not _branches:
        return None
    dtype = [(name, convert('ROOTCODE', 'NUMPY', _branches[name].type)) for name in branches]
    if include_weight:
        if weight_name not in _branches:
            dtype.append((weight_name, weight_dtype))
        else:
            raise ValueError("Weight name '%s' conflicts "
                             "with another field name" % weight_name)
    total_entries = sum([tree.GetEntries() for tree in trees])
    array = np.recarray(shape=(total_entries,), dtype=dtype)
    i = 0
    for tree in trees:
        tree.use_cache(use_cache, cache_size=cache_size, learn_entries=1)
        if use_cache:
            tree.always_read(branches)
        tree_weight = tree.GetWeight()
        for entry in tree:
            for j, branch in enumerate(branches):
                array[i][j] = entry[branch].value
            if include_weight:
                array[i][-1] = tree_weight
            i += 1
    return array
