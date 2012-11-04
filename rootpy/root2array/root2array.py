"""
This module includes:
* conversion of TTrees into NumPy ndarrays and recarrays.
* utility functions for working with ndarrays and recarrays.
* TODO: conversion of TTrees into carrays (http://pypi.python.org/pypi/carray).
"""
import ROOT

import numpy as np
from numpy.lib import recfunctions

from ..types import Variable, convert
from ..utils import asrootpy
from .root_numpy import tree2rec, tree2array


__all__ = [
    'recarray_to_ndarray',
    'tree_to_ndarray',
    'tree_to_recarray',
    'tree_to_recarray_py',
]


def recarray_to_ndarray(rec, fields=None, dtype=np.float32):
    """
    Convert a numpy.recarray into a numpy.ndarray
    """
    if fields is None:
        fields = rec.dtype.names
    ndarray = np.empty((len(rec), len(rec.dtype)), dtype=dtype)
    for idx, field in enumerate(fields):
        ndarray[:, idx] = rec[field]
    return ndarray


def _add_weight_field(arr, tree, include_weight=False,
                      weight_name='weight',
                      weight_dtype='f4'):
    if not include_weight:
        return arr
    weights = np.ones(arr.shape[0], dtype=weight_dtype)
    weights *= tree.GetWeight()
    return recfunctions.rec_append_fields(arr, names=weight_name,
                                          data=weights,
                                          dtypes=weight_dtype)


def _add_weight_column(arr, tree, include_weight=False,
                       weight_dtype='f4'):

    if not include_weight:
        return arr
    weights = np.ones(arr.shape[0], dtype=weight_dtype)
    weights *= tree.GetWeight()
    weights = weights.reshape((arr.shape[0], 1))
    return np.append(arr, weights, axis=1)


def tree_to_ndarray(trees, branches=None,
                    dtype=np.float32,
                    include_weight=False,
                    weight_dtype='f4'):
    """
    Convert a tree or a list of trees into a numpy.ndarray
    """
    if isinstance(trees, (list, tuple)):
        return np.concatenate([
            _add_weight_column(
                recarray_to_ndarray(tree2array(tree, branches),
                                    dtype=dtype),
                tree, include_weight,
                weight_dtype)
            for tree in trees])
    return _add_weight_column(
                recarray_to_ndarray(tree2array(trees, branches),
                                    dtype=dtype),
                trees, include_weight,
                weight_dtype)


def tree_to_recarray(trees, branches=None,
                     include_weight=False,
                     weight_name='weight',
                     weight_dtype='f4'):
    """
    Convert a tree or a list of trees into a numpy.recarray
    with fields corresponding to the tree branches
    """
    if isinstance(trees, (list, tuple)):
        return np.concatenate([
            _add_weight_field(tree2rec(tree, branches),
                              tree, include_weight,
                              weight_name, weight_dtype)
            for tree in trees])
    return _add_weight_field(tree2rec(trees, branches),
                             trees, include_weight,
                             weight_name, weight_dtype)


def tree_to_recarray_py(trees, branches=None,
                        use_cache=False, cache_size=1000000,
                        include_weight=False,
                        weight_name='weight',
                        weight_dtype='f4'):
    """
    Convert a tree or a list of trees into a numpy.recarray
    with fields corresponding to the tree branches

    (the slow pure-Python way...)
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
    dtype = [(name, convert('ROOTCODE', 'NUMPY', _branches[name].type))
             for name in branches]
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
