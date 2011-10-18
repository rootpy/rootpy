"""
This module should handle:
* conversion of TTrees into NumPy arrays. Done
* conversion of a single TBranch into a Numpy array. TODO
* conversion of TTrees into carrays (http://pypi.python.org/pypi/carray). TODO
"""

from .types import Variable, convert
from .utils import asrootpy
import numpy as np


def to_numpy_array(trees, branches=None,
                   use_cache=False, cache_size=1000000,
                   include_weight=False,
                   weight_dtype='f4'):
    
    if type(trees) not in (list, tuple):
        trees = [trees]
    trees = [asrootpy(tree) for tree in trees]
    # if branches is None then select only branches with basic types
    # i.e. no vectors or other special objects
    tree = trees[0]
    _branches = {}
    if branches is None:
        for name, value in tree.buffer.items():
            if isinstance(value, Variable):
                _branches[name] = value
    else:
        if len(set(branches)) != len(branches):
            raise ValueError('branches contains duplicates')
        for branch in branches:
            if branch not in tree.buffer:
                raise ValueError("Branch %s does not exist in tree" % branch)
            value = tree.buffer[branch]
            if not isinstance(value, Variable):
                raise TypeError("Branch %s is not a basic type: %s" % 
                                (branch, type(value)))
            _branches[branch] = tree.buffer[branch]
    if not _branches:
        return None
    dtype = [(name, convert('ROOTCODE', 'NUMPY', value.type)) for name, value in _branches.items()]
    if include_weight:
        if 'weight' not in _branches.keys():
            dtype.append(('weight', weight_dtype))
        else:
            include_weight = False
    total_entries = sum([tree.GetEntries() for tree in trees])
    array = np.recarray(shape=(total_entries,), dtype=dtype)
    i = 0
    for tree in trees:
        tree.use_cache(use_cache, cache_size=cache_size, learn_entries=1)
        if use_cache:
            tree.always_read(_branches.keys())
        weight = tree.GetWeight()
        for entry in tree:
            for j, branch in enumerate(_branches.keys()):
                array[i][j] = entry[branch].value
            if include_weight:
                array[i][-1] = weight
            i += 1
    return array
