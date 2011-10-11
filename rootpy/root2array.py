"""
This module should handle:
* conversion of ROOT's TFile and contained TTrees into HDF5 format with PyTables
  A first attempt is in scripts/root2hd5
* conversion of TTrees into NumPy arrays
* conversion of TTrees into carrays
"""
from .types import Variable, convert
import numpy as np

def to_numpy_array(tree, branches=None, use_cache=False, cache_size=1000000):
    
    # if branches is None then select only branches with basic types
    # i.e. no vectors or other special objects
    _branches = []
    if branches is None:
        for name, value in tree.buffer.items():
            if isinstance(value, Variable):
                _branches.append((name, value))
    else:
        for branch in branches:
            if branch not in tree.buffer:
                raise ValueError("Branch %s does not exist in tree" % branch)
            value = tree.buffer[branch]
            if not isinstance(value, Variable):
                raise TypeError("Branch %s is not a basic type: %s" % 
                                (branch, type(value)))
            _branches.append((branch, tree.buffer[branch]))    
    if not _branches:
        return None
    dtype = [(name, convert('ROOTCODE', 'NUMPY', value.type)) for name, value in _branches]
    array = np.recarray(shape=(tree.GetEntries(),), dtype=dtype)
    tree.use_cache(use_cache, cache_size=cache_size, learn_entries=1)
    if use_cache:
        tree.always_read([name for name, value in _branches])
    for i, entry in enumerate(tree):
        for j, (branch, value) in enumerate(_branches):
            array[i][j] = value.value
    return array
