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
from .. import asrootpy, log; log = log[__name__]

try:
    from root_numpy import tree2rec, tree2array
except ImportError:
    log.critical("root_numpy is needed for root2array. Is it installed and "
                 "importable?")
    raise

__all__ = [
    'recarray_to_ndarray',
    'tree_to_ndarray',
    'tree_to_recarray',
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
