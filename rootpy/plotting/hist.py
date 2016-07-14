# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
from __future__ import absolute_import

from array import array
from math import sqrt
from itertools import product
import operator
import numbers
try:
    from itertools import izip as zip
except ImportError: # will be 3.x series
    pass

import ROOT

from .. import asrootpy, QROOT, log; log = log[__name__]
from ..extern.six.moves import range
from ..base import NamedObject, NamelessConstructorObject
from ..decorators import snake_case_methods, cached_property
from ..context import invisible_canvas
from ..utils.extras import izip_exact
from ..extern.shortuuid import uuid
from .base import Plottable, dim
from .graph import Graph, _Graph1DBase


__all__ = [
    'Hist',
    'Hist1D',
    'Hist2D',
    'Hist3D',
    'HistStack',
    'Efficiency',
    'histogram',
]


def canonify_slice(s, n):
    """
    Convert a slice object into a canonical form
    to simplify treatment in histogram bin content
    and edge slicing.
    """
    if isinstance(s, (int, long)):
        return canonify_slice(slice(s, s + 1, None), n)
    start = s.start % n if s.start is not None else 0
    stop = s.stop % n if s.stop is not None else n
    step = s.step if s.step is not None else 1
    return slice(start, stop, step)


def bin_to_edge_slice(s, n):
    """
    Convert a bin slice into a bin edge slice.
    """
    s = canonify_slice(s, n)
    start = s.start
    stop = s.stop
    if start > stop:
        _stop = start + 1
        start = stop + 1
        stop = _stop
    start = max(start - 1, 0)
    step = abs(s.step)
    if stop <= 1 or start >= n - 1 or stop == start + 1:
        return slice(0, None, min(step, n - 2))
    s = slice(start, stop, abs(s.step))
    if len(range(*s.indices(n - 1))) < 2:
        return slice(start, stop, stop - start - 1)
    return s


class _HistViewBase(object):

    @staticmethod
    def _slice_repr(s):
        if isinstance(s, slice):
            return '[start={0}, stop={1}, step={2}]'.format(
                s.start, s.stop, s.step)
        else:
            return '{0}'.format(s)


class HistIndexView(_HistViewBase):

    def __init__(self, hist, idx):
        if idx.step is not None and abs(idx.step) != 1:
            raise ValueError(
                "rebinning using the global histogram bin "
                "indices is not supported")
        self.hist = hist
        self.idx = idx

    def __iter__(self):
        return self.hist.bins(idx=self.idx, overflow=True)

    def __repr__(self):
        return '{0}({1}, idx={2})'.format(
            self.__class__.__name__, self.hist, self._slice_repr(self.idx))


class HistView(_HistViewBase):

    def __init__(self, hist, x):
        if isinstance(x, slice) and x.step == 0:
            raise ValueError("rebin cannot be zero")
        self.hist = hist
        self.x = x

    @cached_property
    def xedges(self):
        return list(self.hist.xedges())[
            bin_to_edge_slice(self.x, self.hist.nbins(axis=0, overflow=True))]

    @property
    def points(self):
        return self.hist.bins_xyz(ix=self.x, proxy=False)

    def __iter__(self):
        return self.hist.bins_xyz(ix=self.x)

    def __repr__(self):
        return '{0}({1}, x={2})'.format(
            self.__class__.__name__, self.hist, self._slice_repr(self.x))


class Hist2DView(_HistViewBase):

    def __init__(self, hist, x, y):
        if isinstance(x, slice) and x.step == 0:
            raise ValueError("rebin along x cannot be zero")
        if isinstance(y, slice) and y.step == 0:
            raise ValueError("rebin along y cannot be zero")
        self.hist = hist
        self.x = x
        self.y = y

    @cached_property
    def xedges(self):
        return list(self.hist.xedges())[
            bin_to_edge_slice(self.x, self.hist.nbins(axis=0, overflow=True))]

    @cached_property
    def yedges(self):
        return list(self.hist.yedges())[
            bin_to_edge_slice(self.y, self.hist.nbins(axis=1, overflow=True))]

    @property
    def points(self):
        return self.hist.bins_xyz(ix=self.x, iy=self.y, proxy=False)

    def __iter__(self):
        return self.hist.bins_xyz(ix=self.x, iy=self.y)

    def __repr__(self):
        return '{0}({1}, x={2}, y={3})'.format(
            self.__class__.__name__, self.hist,
            self._slice_repr(self.x),
            self._slice_repr(self.y))


class Hist3DView(_HistViewBase):

    def __init__(self, hist, x, y, z):
        if isinstance(x, slice) and x.step == 0:
            raise ValueError("rebin along x cannot be zero")
        if isinstance(y, slice) and y.step == 0:
            raise ValueError("rebin along y cannot be zero")
        if isinstance(z, slice) and z.step == 0:
            raise ValueError("rebin along z cannot be zero")
        self.hist = hist
        self.x = x
        self.y = y
        self.z = z

    @cached_property
    def xedges(self):
        return list(self.hist.xedges())[
            bin_to_edge_slice(self.x, self.hist.nbins(axis=0, overflow=True))]

    @cached_property
    def yedges(self):
        return list(self.hist.yedges())[
            bin_to_edge_slice(self.y, self.hist.nbins(axis=1, overflow=True))]

    @cached_property
    def zedges(self):
        return list(self.hist.zedges())[
            bin_to_edge_slice(self.z, self.hist.nbins(axis=2, overflow=True))]

    @property
    def points(self):
        return self.hist.bins_xyz(ix=self.x, iy=self.y, iz=self.z, proxy=False)

    def __iter__(self):
        return self.hist.bins_xyz(ix=self.x, iy=self.y, iz=self.z)

    def __repr__(self):
        return '{0}({1}, x={2}, y={3}, z={4})'.format(
            self.__class__.__name__, self.hist,
            self._slice_repr(self.x),
            self._slice_repr(self.y),
            self._slice_repr(self.z))


class BinProxy(object):

    def __init__(self, hist, idx):
        self.hist = hist
        self.idx = idx
        self._sum_w2 = hist.GetSumw2()

    @property
    def xyz(self):
        return self.hist.xyz(self.idx)

    @cached_property
    def overflow(self):
        """
        Returns true if this BinProxy is for an overflow bin
        """
        indices = self.hist.xyz(self.idx)
        for i in range(self.hist.GetDimension()):
            if indices[i] == 0 or indices[i] == self.hist.nbins(i) + 1:
                return True
        return False

    @property
    def x(self):
        return self.hist.axis_bininfo(0, self.xyz[0])

    @property
    def y(self):
        return self.hist.axis_bininfo(1, self.xyz[1])

    @property
    def z(self):
        return self.hist.axis_bininfo(2, self.xyz[2])

    @property
    def value(self):
        return self.hist.GetBinContent(self.idx)

    @value.setter
    def value(self, v):
        return self.hist.SetBinContent(self.idx, v)

    @property
    def error(self):
        return self.hist.GetBinError(self.idx)

    @error.setter
    def error(self, e):
        return self.hist.SetBinError(self.idx, e)

    @property
    def sum_w2(self):
        return self._sum_w2.At(self.idx)

    @sum_w2.setter
    def sum_w2(self, w):
        self._sum_w2.SetAt(w, self.idx)

    @property
    def effective_entries(self):
        """
        Number of effective entries in this bin.
        The number of unweighted entries this bin would need to
        contain in order to have the same statistical power as this
        bin with possibly weighted entries, estimated by:

            (sum of weights) ** 2 / (sum of squares of weights)

        """
        sum_w2 = self.sum_w2
        if sum_w2 == 0:
            return abs(self.value)
        return (self.value ** 2) / sum_w2

    def __iadd__(self, other):
        self.value += other.value
        self.sum_w2 += other.sum_w2
        return self

    def __imul__(self, v):
        self.value *= v
        self.error *= v
        return self

    def __idiv__(self, v):
        self.value /= v
        self.error /= v
        return self

    def __ipow__(self, v):
        cur_value = self.value
        if cur_value == 0:
            return self
        new_value = cur_value ** v
        self.value = new_value
        self.error *= new_value / cur_value
        return self

    def __repr__(self):
        return '{0}({1}, {2})'.format(
            self.__class__.__name__, self.hist, self.idx)


class _HistBase(Plottable, NamedObject):

    TYPES = dict(
        (c, [getattr(QROOT, "TH{0}{1}".format(d, c)) for d in (1, 2, 3)])
            for c in "CSIFD")

    def _parse_args(self, args, ignore_extras=False):
        params = [{
            'bins': None,
            'nbins': None,
            'low': None,
            'high': None} for _ in range(dim(self))]

        for param in params:
            if len(args) == 0:
                raise TypeError("did not receive expected number of arguments")
            if hasattr(args[0], '__iter__'):
                edges = list(args[0])
                if len(edges) < 2:
                    raise ValueError(
                        "specify at least two bin edges")
                if sorted(edges) != edges:
                    raise ValueError(
                        "bin edges must be sorted in ascending order")
                if len(set(args[0])) != len(args[0]):
                    raise ValueError("bin edges must not be repeated")
                param['bins'] = args[0]
                param['nbins'] = len(args[0]) - 1
                args = args[1:]
            elif len(args) >= 3:
                nbins = args[0]
                if not isinstance(nbins, int):
                    raise TypeError(
                        "number of bins must be an integer")
                if nbins < 1:
                    raise ValueError(
                        "number of bins must be positive")
                low = args[1]
                if not isinstance(low, numbers.Real):
                    raise TypeError(
                        "lower bound must be an int, float, or long")
                high = args[2]
                if not isinstance(high, numbers.Real):
                    raise TypeError(
                        "upper bound must be an int, float, or long")
                param['nbins'] = nbins
                param['low'] = low
                param['high'] = high
                if low >= high:
                    raise ValueError(
                        "upper bound (you gave {0:f}) "
                        "must be greater than lower "
                        "bound (you gave {1:f})".format(
                            float(low), float(high)))
                args = args[3:]
            else:
                raise TypeError(
                    "did not receive expected number of arguments")

        if ignore_extras:
            # used by Profile where range of profiled axis may be specified
            return params, args

        if len(args) != 0:
            raise TypeError(
                "did not receive expected number of arguments")

        return params

    def xyz(self, idx):
        """
        return binx, biny, binz corresponding to the global bin number
        """
        # Not implemented for Python 3:
        # GetBinXYZ(i, x, y, z)
        nx  = self.GetNbinsX() + 2
        ny  = self.GetNbinsY() + 2
        ndim = self.GetDimension()
        if ndim < 2:
            binx = idx % nx
            biny = 0
            binz = 0
        elif ndim < 3:
            binx = idx % nx
            biny = ((idx - binx) // nx) % ny
            binz = 0
        elif ndim < 4:
            binx = idx % nx
            biny = ((idx - binx) // nx) % ny
            binz = ((idx - binx) // nx - biny) // ny
        else:
            raise NotImplementedError
        return binx, biny, binz

    def axis_bininfo(self, axi, i):
        class bi:
            axis = self.axis(axi)
            low = axis.GetBinLowEdge(i)
            center = axis.GetBinCenter(i)
            high = axis.GetBinUpEdge(i)
            width = axis.GetBinWidth(i)
        return bi

    def bins(self, idx=None, overflow=False):
        if idx is None:
            idx = range(self.GetSize())
        elif isinstance(idx, slice):
            idx = range(*idx.indices(self.GetSize()))
            overflow = True
        else:
            idx = [self._range_check(idx)]
            overflow = True
        for i in idx:
            bproxy = BinProxy(self, i)
            if not overflow and bproxy.overflow:
                continue
            yield bproxy

    def bins_xyz(self, ix, iy=0, iz=0, proxy=True):
        xl = self.nbins(axis=0, overflow=True)
        yl = self.nbins(axis=1, overflow=True)
        zl = self.nbins(axis=2, overflow=True)
        if isinstance(ix, slice):
            ix = range(*ix.indices(xl))
        else:
            ix = [self._range_check(ix, axis=0)]
        if isinstance(iy, slice):
            iy = range(*iy.indices(yl))
        else:
            iy = [self._range_check(iy, axis=1)]
        if isinstance(iz, slice):
            iz = range(*iz.indices(zl))
        else:
            iz = [self._range_check(iz, axis=2)]
        if proxy:
            for x, y, z in product(ix, iy, iz):
                yield BinProxy(self, xl * yl * z + xl * y + x)
        else:
            for point in product(ix, iy, iz):
                yield point

    @classmethod
    def divide(cls, h1, h2, c1=1., c2=1., option='', fill_value=None):
        ratio = h1.Clone()
        ROOT.TH1.Divide(ratio, h1, h2, c1, c2, option)
        if fill_value is not None:
            for ratiobin, h2bin in zip(ratio.bins(), h2.bins()):
                if h2bin.value == 0:
                    ratiobin.value = fill_value
        return ratio

    def nbins(self, axis=0, overflow=False):
        """
        Get the number of bins along an axis
        """
        if axis == 0:
            nbins = self.GetNbinsX()
        elif axis == 1:
            nbins = self.GetNbinsY()
        elif axis == 2:
            nbins = self.GetNbinsZ()
        else:
            raise ValueError("axis must be 0, 1, or 2")
        if overflow:
            nbins += 2
        return nbins

    def bins_range(self, axis=0, overflow=False):
        """
        Return a range of bin indices for iterating along an axis

        Parameters
        ----------

        axis : int, optional (default=1)
            The axis (0, 1 or 2).

        overflow : bool, optional (default=False)
            If True then include the underflow and overflow bins
            otherwise only include the visible bins.

        Returns
        -------

        an range object of bin indices

        """
        nbins = self.nbins(axis=axis, overflow=False)
        if overflow:
            start = 0
            end_offset = 2
        else:
            start = 1
            end_offset = 1
        return range(start, nbins + end_offset)

    @property
    def axes(self):
        return [self.axis(i) for i in range(self.GetDimension())]

    def axis(self, axis=0):
        if axis == 0:
            return self.GetXaxis()
        elif axis == 1:
            return self.GetYaxis()
        elif axis == 2:
            return self.GetZaxis()
        else:
            raise ValueError("axis must be 0, 1, or 2")

    @property
    def entries(self):
        return self.GetEntries()

    @entries.setter
    def entries(self, value):
        self.SetEntries(value)

    def __len__(self):
        """
        The total number of bins, including overflow bins
        """
        return self.GetSize()

    def __iter__(self):
        """
        Iterate over the bin proxies
        """
        return self.bins(overflow=True)

    def _range_check(self, index, axis=None):

        if axis is None:
            size = self.GetSize()
        else:
            size = self.nbins(axis=axis)
            if axis < self.GetDimension():
                size += 2
        try:
            if index < 0:
                if index < - size:
                    raise IndexError
                return index % size
            elif index >= size:
                raise IndexError
        except IndexError:
            if axis is None:
                raise IndexError(
                    "global bin index {0:d} is out of range".format(index))
            else:
                raise IndexError(
                    "bin index {0:d} along axis {1:d} is out of range".format(
                        index, axis))
        return index

    def GetBin(self, ix, iy=0, iz=0):
        ix = self._range_check(ix, axis=0)
        iy = self._range_check(iy, axis=1)
        iz = self._range_check(iz, axis=2)
        return super(_HistBase, self).GetBin(ix, iy, iz)

    def __getitem__(self, index):
        """
        Return a BinProxy or list of BinProxies if index is a slice.
        """
        if isinstance(index, slice):
            if isinstance(self, _Hist):
                return HistView(self, index)
            return HistIndexView(self, index)
        if isinstance(index, tuple):
            ix, iy, iz = 0, 0, 0
            ndim = self.GetDimension()
            view = False
            if ndim == 2:
                try:
                    ix, iy = index
                except ValueError:
                    raise IndexError(
                        "must index along only two "
                        "axes of a 2D histogram")
                if isinstance(ix, slice) or isinstance(iy, slice):
                    return Hist2DView(self, x=ix, y=iy)
            elif ndim == 3:
                try:
                    ix, iy, iz = index
                except ValueError:
                    raise IndexError(
                        "must index along exactly three "
                        "axes of a 3D histogram")
                if (isinstance(ix, slice) or isinstance(iy, slice)
                        or isinstance(iz, slice)):
                    return Hist3DView(self, x=ix, y=iy, z=iz)
            else:
                raise IndexError(
                    "must index along only one "
                    "axis of a 1D histogram")
            index = self.GetBin(ix, iy, iz)
        else:
            index = self._range_check(index)
        return BinProxy(self, index)

    def __setitem__(self, index, value):
        """
        Set bin contents and additionally bin errors if value is a BinProxy or
        a 2-tuple containing the value and error.
        If index is a slice then value must be a list of values, BinProxies, or
        2-tuples of the same length as the slice.
        """
        is_slice = isinstance(index, slice)
        is_tuple = (not is_slice) and isinstance(index, tuple)
        if is_slice or is_tuple:

            if isinstance(value, _HistBase):
                self[index] = value[index]
                return

            if is_slice:
                indices = range(*index.indices(self.GetSize()))

            else:
                ndim = self.GetDimension()
                xl = self.nbins(0, overflow=True)
                yl = self.nbins(1, overflow=True)
                if ndim == 2:
                    try:
                        ix, iy = index
                    except ValueError:
                        raise IndexError(
                            "must index along only two "
                            "axes of a 2D histogram")
                    if isinstance(ix, slice):
                        ix = range(*ix.indices(xl))
                    else:
                        ix = [self._range_check(ix, axis=0)]
                    if isinstance(iy, slice):
                        iy = range(*iy.indices(yl))
                    else:
                        iy = [self._range_check(iy, axis=1)]
                    iz = [0]
                elif ndim == 3:
                    try:
                        ix, iy, iz = index
                    except ValueError:
                        raise IndexError(
                            "must index along exactly three "
                            "axes of a 3D histogram")
                    if isinstance(ix, slice):
                        ix = range(*ix.indices(xl))
                    else:
                        ix = [self._range_check(ix, axis=0)]
                    if isinstance(iy, slice):
                        iy = range(*iy.indices(yl))
                    else:
                        iy = [self._range_check(iy, axis=1)]
                    if isinstance(iz, slice):
                        iz = range(*iz.indices(self.nbins(2, overflow=True)))
                    else:
                        iz = [self._range_check(iz, axis=2)]
                else:
                    raise IndexError(
                        "must index along only one "
                        "axis of a 1D histogram")
                indices = (xl * yl * z + xl * y + x
                    for (x, y, z) in product(ix, iy, iz))

            if isinstance(value, _HistViewBase):
                for i, v in izip_exact(indices, value):
                    self.SetBinContent(i, v.value)
                    self.SetBinError(i, v.error)
            elif hasattr(value, '__iter__') and not isinstance(value, tuple):
                if value and isinstance(value[0], tuple):
                    for i, v in izip_exact(indices, value):
                        _value, _error = value
                        self.SetBinContent(i, _value)
                        self.SetBinError(i, _error)
                else:
                    for i, v in izip_exact(indices, value):
                        self.SetBinContent(i, v)
            elif isinstance(value, BinProxy):
                v, e = value.value, value.error
                for i in indices:
                    self.SetBinContent(i, v)
                    self.SetBinError(i, e)
            elif isinstance(value, tuple):
                _value, _error = value
                for i in indices:
                    self.SetBinContent(i, _value)
                    self.SetBinError(i, _error)
            else:
                for i in indices:
                    self.SetBinContent(i, value)
            return

        index = self._range_check(index)

        if isinstance(value, BinProxy):
            self.SetBinContent(index, value.value)
            self.SetBinError(index, value.error)
        elif isinstance(value, tuple):
            value, error = value
            self.SetBinContent(index, value)
            self.SetBinError(index, error)
        else:
            self.SetBinContent(index, value)

    def uniform(self, axis=None, precision=1E-7):
        """
        Return True if the binning is uniform along the specified axis.
        If axis is None (the default), then return True if the binning is
        uniform along all axes. Otherwise return False.

        Parameters
        ----------

        axis : int (default=None)
            Axis along which to check if the binning is uniform. If None,
            then check all axes.

        precision : float (default=1E-7)
            The threshold below which differences in bin widths are ignored and
            treated as equal.

        Returns
        -------

        True if the binning is uniform, otherwise False.

        """
        if axis is None:
            for axis in range(self.GetDimension()):
                widths = list(self._width(axis=axis))
                if not all(abs(x - widths[0]) < precision for x in widths):
                    return False
            return True
        widths = list(self._width(axis=axis))
        return all(abs(x - widths[0]) < precision for x in widths)

    def uniform_binned(self, name=None):
        """
        Return a new histogram with constant width bins along all axes by
        using the bin indices as the bin edges of the new histogram.
        """
        if self.GetDimension() == 1:
            new_hist = Hist(
                self.GetNbinsX(), 0, self.GetNbinsX(),
                name=name, type=self.TYPE)
        elif self.GetDimension() == 2:
            new_hist = Hist2D(
                self.GetNbinsX(), 0, self.GetNbinsX(),
                self.GetNbinsY(), 0, self.GetNbinsY(),
                name=name, type=self.TYPE)
        else:
            new_hist = Hist3D(
                self.GetNbinsX(), 0, self.GetNbinsX(),
                self.GetNbinsY(), 0, self.GetNbinsY(),
                self.GetNbinsZ(), 0, self.GetNbinsZ(),
                name=name, type=self.TYPE)
        # copy over the bin contents and errors
        for outbin, inbin in zip(new_hist.bins(), self.bins()):
            outbin.value = inbin.value
            outbin.error = inbin.error
        new_hist.decorate(self)
        new_hist.entries = self.entries
        return new_hist

    def underflow(self, axis=0):
        """
        Return the underflow for the given axis.

        Depending on the dimension of the histogram, may return an array.
        """
        if axis not in range(3):
            raise ValueError("axis must be 0, 1, or 2")
        if self.DIM == 1:
            return self.GetBinContent(0)
        elif self.DIM == 2:
            def idx(i):
                arg = [i]
                arg.insert(axis, 0)
                return arg
            return [
                self.GetBinContent(*idx(i))
                for i in self.bins_range(axis=(axis + 1) % 2, overflow=True)]
        elif self.DIM == 3:
            axes = [0, 1, 2]
            axes.remove(axis)
            axis2, axis3 = axes
            def idx(i, j):
                arg = [i, j]
                arg.insert(axis, 0)
                return arg
            return [[
                self.GetBinContent(*idx(i, j))
                for i in self.bins_range(axis=axis2, overflow=True)]
                for j in self.bins_range(axis=axis3, overflow=True)]

    def overflow(self, axis=0):
        """
        Return the overflow for the given axis.

        Depending on the dimension of the histogram, may return an array.
        """
        if axis not in range(3):
            raise ValueError("axis must be 0, 1, or 2")
        if self.DIM == 1:
            return self.GetBinContent(self.nbins(0) + 1)
        elif self.DIM == 2:
            axes = [0, 1]
            axes.remove(axis)
            axis2 = axes[0]
            nbins_axis = self.nbins(axis)
            def idx(i):
                arg = [i]
                arg.insert(axis, nbins_axis + 1)
                return arg
            return [
                self.GetBinContent(*idx(i))
                for i in self.bins_range(axis=axis2, overflow=True)]
        elif self.DIM == 3:
            axes = [0, 1, 2]
            axes.remove(axis)
            axis2, axis3 = axes
            nbins_axis = self.nbins(axis)
            def idx(i, j):
                arg = [i, j]
                arg.insert(axis, nbins_axis + 1)
                return arg
            return [[
                self.GetBinContent(*idx(i, j))
                for i in self.bins_range(axis=axis2, overflow=True)]
                for j in self.bins_range(axis=axis3, overflow=True)]

    def lowerbound(self, axis=0):
        """
        Get the lower bound of the binning along an axis
        """
        if not 0 <= axis < self.GetDimension():
            raise ValueError(
                "axis must be a non-negative integer less than "
                "the dimensionality of the histogram")
        if axis == 0:
            return self.xedges(1)
        if axis == 1:
            return self.yedges(1)
        if axis == 2:
            return self.zedges(1)
        raise TypeError("axis must be an integer")

    def upperbound(self, axis=0):
        """
        Get the upper bound of the binning along an axis
        """
        if not 0 <= axis < self.GetDimension():
            raise ValueError(
                "axis must be a non-negative integer less than "
                "the dimensionality of the histogram")
        if axis == 0:
            return self.xedges(-2)
        if axis == 1:
            return self.yedges(-2)
        if axis == 2:
            return self.zedges(-2)
        raise TypeError("axis must be an integer")

    def bounds(self, axis=0):
        """
        Get the lower and upper bounds of the binning along an axis
        """
        if not 0 <= axis < self.GetDimension():
            raise ValueError(
                "axis must be a non-negative integer less than "
                "the dimensionality of the histogram")
        if axis == 0:
            return self.xedges(1), self.xedges(-2)
        if axis == 1:
            return self.yedges(1), self.yedges(-2)
        if axis == 2:
            return self.zedges(1), self.zedges(-2)
        raise TypeError("axis must be an integer")

    def _centers(self, axis, index=None, overflow=False):
        nbins = self.nbins(axis)
        ax = self.axis(axis)
        if index is None:
            def temp_generator():
                if overflow:
                    yield float('-inf')
                for index in range(1, nbins + 1):
                    yield ax.GetBinCenter(index)
                if overflow:
                    yield float('+inf')
            return temp_generator()
        index = index % (nbins + 2)
        if index == 0:
            return float('-inf')
        elif index == nbins + 1:
            return float('+inf')
        return ax.GetBinCenter(index)

    def _edgesl(self, axis, index=None, overflow=False):
        nbins = self.nbins(axis)
        ax = self.axis(axis)
        if index is None:
            def temp_generator():
                if overflow:
                    yield float('-inf')
                for index in range(1, nbins + 1):
                    yield ax.GetBinLowEdge(index)
                if overflow:
                    yield ax.GetBinUpEdge(index)
            return temp_generator()
        index = index % (nbins + 2)
        if index == 0:
            return float('-inf')
        return ax.GetBinLowEdge(index)

    def _edgesh(self, axis, index=None, overflow=False):
        nbins = self.nbins(axis)
        ax = self.axis(axis)
        if index is None:
            def temp_generator():
                if overflow:
                    yield ax.GetBinUpEdge(0)
                for index in range(1, nbins + 1):
                    yield ax.GetBinUpEdge(index)
                if overflow:
                    yield float('+inf')
            return temp_generator()
        index = index % (nbins + 2)
        if index == nbins + 1:
            return float('+inf')
        return ax.GetBinUpEdge(index)

    def _edges(self, axis, index=None, overflow=False):
        nbins = self.nbins(axis)
        ax = self.axis(axis)
        if index is None:
            def temp_generator():
                if overflow:
                    yield float('-inf')
                for index in range(1, nbins + 1):
                    yield ax.GetBinLowEdge(index)
                yield ax.GetBinUpEdge(nbins)
                if overflow:
                    yield float('+inf')
            return temp_generator()
        index = index % (nbins + 3)
        if index == 0:
            return float('-inf')
        if index == nbins + 2:
            return float('+inf')
        return ax.GetBinLowEdge(index)

    def _width(self, axis, index=None, overflow=False):
        nbins = self.nbins(axis)
        ax = self.axis(axis)
        if index is None:
            def temp_generator():
                if overflow:
                    yield float('+inf')
                for index in range(1, nbins + 1):
                    yield ax.GetBinWidth(index)
                if overflow:
                    yield float('+inf')
            return temp_generator()
        index = index % (nbins + 2)
        if index in (0, nbins + 1):
            return float('+inf')
        return ax.GetBinWidth(index)

    def _erravg(self, axis, index=None, overflow=False):
        nbins = self.nbins(axis)
        ax = self.axis(axis)
        if index is None:
            def temp_generator():
                if overflow:
                    yield float('+inf')
                for index in range(1, nbins + 1):
                    yield ax.GetBinWidth(index) / 2.
                if overflow:
                    yield float('+inf')
            return temp_generator()
        index = index % (nbins + 2)
        if index in (0, nbins + 1):
            return float('+inf')
        return ax.GetBinWidth(index) / 2.

    def _err(self, axis, index=None, overflow=False):
        nbins = self.nbins(axis)
        ax = self.axis(axis)
        if index is None:
            def temp_generator():
                if overflow:
                    yield (float('+inf'), float('+inf'))
                for index in range(1, nbins + 1):
                    w = ax.GetBinWidth(index) / 2.
                    yield (w, w)
                if overflow:
                    yield (float('+inf'), float('+inf'))
            return temp_generator()
        index = index % (nbins + 2)
        if index in (0, nbins + 1):
            return (float('+inf'), float('+inf'))
        w = ax.GetBinWidth(index) / 2.
        return (w, w)

    def check_compatibility(self, other, check_edges=False, precision=1E-7):
        """
        Test whether two histograms are considered compatible by the number of
        dimensions, number of bins along each axis, and optionally the bin
        edges.

        Parameters
        ----------

        other : histogram
            A rootpy histogram

        check_edges : bool, optional (default=False)
            If True then also check that the bin edges are equal within
            the specified precision.

        precision : float, optional (default=1E-7)
            The value below which differences between floats are treated as
            nil when comparing bin edges.

        Raises
        ------

        TypeError
            If the histogram dimensionalities do not match

        ValueError
            If the histogram sizes, number of bins along an axis, or
            optionally the bin edges do not match

        """
        if self.GetDimension() != other.GetDimension():
            raise TypeError("histogram dimensionalities do not match")
        if len(self) != len(other):
            raise ValueError("histogram sizes do not match")
        for axis in range(self.GetDimension()):
            if self.nbins(axis=axis) != other.nbins(axis=axis):
                raise ValueError(
                    "numbers of bins along axis {0:d} do not match".format(
                        axis))
        if check_edges:
            for axis in range(self.GetDimension()):
                if not all([abs(l - r) < precision
                    for l, r in zip(self._edges(axis), other._edges(axis))]):
                    raise ValueError(
                        "edges do not match along axis {0:d}".format(axis))

    def compatible(self, other, check_edges=False, precision=1E-7):
        try:
            self.check_compatibility(other,
                check_edges=check_edges, precision=precision)
        except (TypeError, ValueError):
            return False
        return True

    def __add__(self, other):
        copy = self.Clone()
        copy += other
        return copy

    def __iadd__(self, other):
        if isinstance(other, numbers.Real):
            if other != 0:
                for bin in self.bins(overflow=True):
                    bin.value += other
        else:
            self.Add(other)
        return self

    def __radd__(self, other):
        if isinstance(other, numbers.Real):
            copy = self.Clone()
            if other != 0:
                copy += other
            return copy
        return NotImplemented

    def __sub__(self, other):
        copy = self.Clone()
        copy -= other
        return copy

    def __isub__(self, other):
        if isinstance(other, numbers.Real):
            if other != 0:
                for bin in self.bins(overflow=True):
                    bin.value -= other
        else:
            self.Add(other, -1.)
        return self

    def __rsub__(self, other):
        if isinstance(other, numbers.Real):
            copy = self.Clone()
            if other != 0:
                for bin in copy.bins(overflow=True):
                    bin.value = other - bin.value
            return copy
        return NotImplemented

    def __mul__(self, other):
        copy = self.Clone()
        copy *= other
        return copy

    def __imul__(self, other):
        if isinstance(other, numbers.Real):
            self.Scale(other)
            return self
        self.Multiply(other)
        return self

    def __rmul__(self, other):
        if isinstance(other, numbers.Real):
            copy = self.Clone()
            if other != 1:
                copy *= other
            return copy
        return NotImplemented

    def __div__(self, other):
        copy = self.Clone()
        copy /= other
        return copy

    __truediv__ = __div__

    def __idiv__(self, other):
        if isinstance(other, numbers.Real):
            if other == 0:
                raise ZeroDivisionError(
                    "attempting to divide histogram by zero")
            self.Scale(1. / other)
            return self
        self.Divide(other)
        return self

    __itruediv__ = __idiv__

    def __rdiv__(self, other):
        if isinstance(other, numbers.Real):
            copy = self.Clone()
            for bin in copy.bins(overflow=True):
                v = bin.value
                if v != 0:
                    bin.value = other / v
            return copy
        return NotImplemented

    __rtruediv__ = __rdiv__

    def __ipow__(self, other, modulo=None):
        if modulo is not None:
            return NotImplemented
        if isinstance(other, numbers.Real):
            for bin in self.bins(overflow=True):
                bin **= other
        elif isinstance(other, _HistBase):
            self.check_compatibility(other)
            for this_bin, other_bin in zip(
                    self.bins(overflow=True),
                    other.bins(overflow=True)):
                this_bin **= other_bin.value
        else:
            return NotImplemented
        return self

    def __pow__(self, other, modulo=None):
        if modulo is not None:
            return NotImplemented
        copy = self.Clone()
        copy **= other
        return copy

    def __cmp__(self, other):
        return cmp(self.Integral(), other.Integral())

    def fill_array(self, array, weights=None):
        """
        Fill this histogram with a NumPy array
        """
        try:
            try:
                from root_numpy import fill_hist as fill_func
            except ImportError:
                from root_numpy import fill_array as fill_func
        except ImportError:
            log.critical(
                "root_numpy is needed for Hist*.fill_array. "
                "Is it installed and importable?")
            raise
        fill_func(self, array, weights=weights)

    def fill_view(self, view):
        """
        Fill this histogram from a view of another histogram
        """
        other = view.hist
        _other_x_center = other.axis(0).GetBinCenter
        _other_y_center = other.axis(1).GetBinCenter
        _other_z_center = other.axis(2).GetBinCenter
        _other_get = other.GetBinContent
        _other_get_bin = super(_HistBase, other).GetBin
        other_sum_w2 = other.GetSumw2()
        _other_sum_w2_at = other_sum_w2.At

        _find = self.FindBin
        sum_w2 = self.GetSumw2()
        _sum_w2_at = sum_w2.At
        _sum_w2_setat = sum_w2.SetAt
        _set = self.SetBinContent
        _get = self.GetBinContent

        for x, y, z in view.points:
            idx = _find(
                _other_x_center(x),
                _other_y_center(y),
                _other_z_center(z))
            other_idx = _other_get_bin(x, y, z)
            _set(idx, _get(idx) + _other_get(other_idx))
            _sum_w2_setat(
                _sum_w2_at(idx) + _other_sum_w2_at(other_idx),
                idx)

    def FillRandom(self, func, ntimes=5000):
        if isinstance(func, QROOT.TF1):
            func = func.GetName()
        super(_HistBase, self).FillRandom(func, ntimes)
        return self

    def get_sum_w2(self, ix, iy=0, iz=0):
        """
        Obtain the true number of entries in the bin weighted by w^2
        """
        if self.GetSumw2N() == 0:
            raise RuntimeError(
                "Attempting to access Sumw2 in histogram "
                "where weights were not stored")
        xl = self.nbins(axis=0, overflow=True)
        yl = self.nbins(axis=1, overflow=True)
        idx = xl * yl * iz + xl * iy + ix
        if not 0 <= idx < self.GetSumw2N():
            raise IndexError("bin index out of range")
        return self.GetSumw2().At(idx)

    def set_sum_w2(self, w, ix, iy=0, iz=0):
        """
        Sets the true number of entries in the bin weighted by w^2
        """
        if self.GetSumw2N() == 0:
            raise RuntimeError(
                "Attempting to access Sumw2 in histogram "
                "where weights were not stored")
        xl = self.nbins(axis=0, overflow=True)
        yl = self.nbins(axis=1, overflow=True)
        idx = xl * yl * iz + xl * iy + ix
        if not 0 <= idx < self.GetSumw2N():
            raise IndexError("bin index out of range")
        self.GetSumw2().SetAt(w, idx)

    def merge_bins(self, bin_ranges, axis=0):
        """
        Merge bins in bin ranges

        Parameters
        ----------

        bin_ranges : list of tuples
            A list of tuples of bin indices for each bin range to be merged
            into one bin.

        axis : int (default=1)
            The integer identifying the axis to merge bins along.

        Returns
        -------

        hist : TH1
            The rebinned histogram.

        Examples
        --------

        Merge the overflow bins into the first and last real bins::

            newhist = hist.merge_bins([(0, 1), (-2, -1)])

        """
        ndim = self.GetDimension()
        if axis > ndim - 1:
            raise ValueError(
                "axis is out of range")
        axis_bins = self.nbins(axis=axis, overflow=True)

        # collect the indices along this axis to be merged
        # support negative indices via slicing
        windows = []
        for window in bin_ranges:
            if len(window) != 2:
                raise ValueError(
                    "bin range tuples must contain two elements")
            l, r = window
            if l == r:
                raise ValueError(
                    "bin indices must not be equal in a merging window")
            if l < 0 and r >= 0:
                raise ValueError(
                    "invalid bin range")
            if r == -1:
                r = axis_bins
            else:
                r += 1
            bin_idx = range(*slice(l, r).indices(axis_bins))
            if bin_idx: # skip []
                windows.append(list(bin_idx))

        if not windows:
            # no merging will take place so return a clone
            return self.Clone()

        # check that windows do not overlap
        if len(windows) > 1:
            full_list = windows[0]
            for window in windows[1:]:
                full_list += window
            if len(full_list) != len(set(full_list)):
                raise ValueError("bin index windows overlap")

        # construct a mapping from old to new bin index along this axis
        windows.sort()
        mapping = {}
        left_idx = {}
        offset = 0
        for window in windows:
            # put underflow in first bin
            new_idx = window[0] - offset or 1
            left_idx[window[0] or 1] = None
            for idx in window:
                mapping[idx] = new_idx
            offset += len(window) - 1
            if window[0] == 0:
                offset -= 1

        new_axis_bins = axis_bins - offset

        # construct new bin edges
        new_edges = []
        for i, edge in enumerate(self._edges(axis)):
            if (i != axis_bins - 2 and i + 1 in mapping
                and i + 1 not in left_idx):
                continue
            new_edges.append(edge)

        # construct new histogram and fill
        new_hist = self.empty_clone(binning=new_edges, axis=axis)

        this_axis = self.axis(axis)
        new_axis = new_hist.axis(axis)

        def translate(idx):
            if idx in mapping:
                return mapping[idx]
            if idx == 0:
                return 0
            # use TH1.FindBin to determine where the bins should be merged
            return new_axis.FindBin(this_axis.GetBinCenter(idx))

        for bin in self.bins(overflow=True):
            xyz = bin.xyz
            new_xyz = list(xyz)
            new_xyz[axis] = translate(int(xyz[axis]))

            x, y, z = new_xyz

            new_v = new_hist.GetBinContent(x, y, z)
            new_hist.SetBinContent(x, y, z, new_v + bin.value)

            sum_w2 = self.get_sum_w2(*xyz)
            new_sum_w2 = new_hist.get_sum_w2(x, y, z)
            new_hist.set_sum_w2(sum_w2 + new_sum_w2, x, y, z)

        # transfer stats info
        stat_array = array('d', [0.] * 10)
        self.GetStats(stat_array)
        new_hist.PutStats(stat_array)
        entries = self.GetEntries()
        new_hist.SetEntries(entries)
        return new_hist

    def rebinned(self, bins, axis=0):
        """
        Return a new rebinned histogram

        Parameters
        ----------

        bins : int, tuple, or iterable
            If ``bins`` is an int, then return a histogram that is rebinned by
            grouping N=``bins`` bins together along the axis ``axis``.
            If ``bins`` is a tuple, then it must contain the same number of
            elements as there are dimensions of this histogram and each element
            will be used to rebin along the associated axis.
            If ``bins`` is another iterable, then it will define the bin
            edges along the axis ``axis`` in the new rebinned histogram.

        axis : int, optional (default=0)
            The axis to rebin along.

        Returns
        -------

        The rebinned histogram

        """
        ndim = self.GetDimension()
        if axis >= ndim:
            raise ValueError(
                "axis must be less than the dimensionality of the histogram")

        if isinstance(bins, int):
            _bins = [1] * ndim
            try:
                _bins[axis] = bins
            except IndexError:
                raise ValueError("axis must be 0, 1, or 2")
            bins = tuple(_bins)

        if isinstance(bins, tuple):
            if len(bins) != ndim:
                raise ValueError(
                    "bins must be a tuple with the same "
                    "number of elements as histogram axes")
            newname = '{0}_{1}'.format(self.__class__.__name__, uuid())
            if ndim == 1:
                hist = self.Rebin(bins[0], newname)
            elif ndim == 2:
                hist = self.Rebin2D(bins[0], bins[1], newname)
            else:
                hist = self.Rebin3D(bins[0], bins[1], bins[2], newname)
            hist = asrootpy(hist)
        elif hasattr(bins, '__iter__'):
            hist = self.empty_clone(bins, axis=axis)
            nbinsx = self.nbins(0)
            nbinsy = self.nbins(1)
            nbinsz = self.nbins(2)
            xaxis = self.xaxis
            yaxis = self.yaxis
            zaxis = self.zaxis
            sum_w2 = self.GetSumw2()
            _sum_w2_at = sum_w2.At
            new_sum_w2 = hist.GetSumw2()
            _new_sum_w2_at = new_sum_w2.At
            _new_sum_w2_setat = new_sum_w2.SetAt
            _x_center = xaxis.GetBinCenter
            _y_center = yaxis.GetBinCenter
            _z_center = zaxis.GetBinCenter
            _find = hist.FindBin
            _set = hist.SetBinContent
            _get = hist.GetBinContent
            _this_get = self.GetBinContent
            _get_bin = super(_HistBase, self).GetBin
            for z in range(1, nbinsz + 1):
                for y in range(1, nbinsy + 1):
                    for x in range(1, nbinsx + 1):
                        newbin = _find(
                            _x_center(x), _y_center(y), _z_center(z))
                        idx = _get_bin(x, y, z)
                        _set(newbin, _get(newbin) + _this_get(idx))
                        _new_sum_w2_setat(
                            _new_sum_w2_at(newbin) + _sum_w2_at(idx),
                            newbin)
            hist.SetEntries(self.GetEntries())
        else:
            raise TypeError(
                "bins must either be an integer, a tuple, or an iterable")
        return hist

    def smoothed(self, iterations=1):
        """
        Return a smoothed copy of this histogram

        Parameters
        ----------

        iterations : int, optional (default=1)
            The number of smoothing iterations

        Returns
        -------

        hist : asrootpy'd histogram
            The smoothed histogram

        """
        copy = self.Clone(shallow=True)
        copy.Smooth(iterations)
        return copy

    def empty_clone(self, binning=None, axis=0, type=None, **kwargs):
        """
        Return a new empty histogram. The binning may be modified
        along one axis by specifying the binning and axis arguments.
        If binning is False, then the corresponding axis is dropped
        from the returned histogram.
        """
        ndim = self.GetDimension()
        if binning is False and ndim == 1:
            raise ValueError(
                "cannot remove the x-axis of a 1D histogram")
        args = []
        for iaxis in range(ndim):
            if iaxis == axis:
                if binning is False:
                    # skip this axis
                    continue
                elif binning is not None:
                    if hasattr(binning, '__iter__'):
                        binning = (binning,)
                    args.extend(binning)
                    continue
            args.append(list(self._edges(axis=iaxis)))
        if type is None:
            type = self.TYPE
        if binning is False:
            ndim -= 1
        cls = [Hist, Hist2D, Hist3D][ndim - 1]
        return cls(*args, type=type, **kwargs)

    def quantiles(self, quantiles,
                  axis=0, strict=False,
                  recompute_integral=False):
        """
        Calculate the quantiles of this histogram.

        Parameters
        ----------

        quantiles : list or int
            A list of cumulative probabilities or an integer used to determine
            equally spaced values between 0 and 1 (inclusive).

        axis : int, optional (default=0)
            The axis to compute the quantiles along. 2D and 3D histograms are
            first projected along the desired axis before computing the
            quantiles.

        strict : bool, optional (default=False)
            If True, then return the sorted unique quantiles corresponding
            exactly to bin edges of this histogram.

        recompute_integral : bool, optional (default=False)
            If this histogram was filled with SetBinContent instead of Fill,
            then the integral must be computed before calculating the
            quantiles.

        Returns
        -------

        output : list or numpy array
            If NumPy is importable then an array of the quantiles is returned,
            otherwise a list is returned.

        """
        if axis >= self.GetDimension():
            raise ValueError(
                "axis must be less than the dimensionality of the histogram")
        if recompute_integral:
            self.ComputeIntegral()
        if isinstance(self, _Hist2D):
            newname = '{0}_{1}'.format(self.__class__.__name__, uuid())
            if axis == 0:
                proj = self.ProjectionX(newname, 1, self.nbins(1))
            elif axis == 1:
                proj = self.ProjectionY(newname, 1, self.nbins(0))
            else:
                raise ValueError("axis must be 0 or 1")
            return asrootpy(proj).quantiles(
                quantiles, strict=strict, recompute_integral=False)
        elif isinstance(self, _Hist3D):
            newname = '{0}_{1}'.format(self.__class__.__name__, uuid())
            if axis == 0:
                proj = self.ProjectionX(
                    newname, 1, self.nbins(1), 1, self.nbins(2))
            elif axis == 1:
                proj = self.ProjectionY(
                    newname, 1, self.nbins(0), 1, self.nbins(2))
            elif axis == 2:
                proj = self.ProjectionZ(
                    newname, 1, self.nbins(0), 1, self.nbins(1))
            else:
                raise ValueError("axis must be 0, 1, or 2")
            return asrootpy(proj).quantiles(
                quantiles, strict=strict, recompute_integral=False)
        try:
            import numpy as np
        except ImportError:
            # use python implementation
            use_numpy = False
        else:
            use_numpy = True
        if isinstance(quantiles, int):
            num_quantiles = quantiles
            if use_numpy:
                qs = np.linspace(0, 1, num_quantiles)
                output = np.empty(num_quantiles, dtype=float)
            else:
                def linspace(start, stop, n):
                    if n == 1:
                        yield start
                        return
                    h = float(stop - start) / (n - 1)
                    for i in range(n):
                        yield start + h * i
                quantiles = list(linspace(0, 1, num_quantiles))
                qs = array('d', quantiles)
                output = array('d', [0.] * num_quantiles)
        else:
            num_quantiles = len(quantiles)
            if use_numpy:
                qs = np.array(quantiles, dtype=float)
                output = np.empty(num_quantiles, dtype=float)
            else:
                qs = array('d', quantiles)
                output = array('d', [0.] * num_quantiles)
        if strict:
            integral = self.GetIntegral()
            nbins = self.nbins(0)
            if use_numpy:
                edges = np.empty(nbins + 1, dtype=float)
                self.GetLowEdge(edges)
                edges[-1] = edges[-2] + self.GetBinWidth(nbins)
                integral = np.ndarray((nbins + 1,), dtype=float, buffer=integral)
                idx = np.searchsorted(integral, qs, side='left')
                output = np.unique(np.take(edges, idx))
            else:
                quantiles = list(set(qs))
                quantiles.sort()
                output = []
                ibin = 0
                for quant in quantiles:
                    # find first bin greater than or equal to quant
                    while integral[ibin] < quant and ibin < nbins + 1:
                        ibin += 1
                    edge = self.GetBinLowEdge(ibin + 1)
                    output.append(edge)
                    if ibin >= nbins + 1:
                        break
                output = list(set(output))
                output.sort()
            return output
        self.GetQuantiles(num_quantiles, output, qs)
        if use_numpy:
            return output
        return list(output)

    def max(self, include_error=False):
        if not include_error:
            return self.GetBinContent(self.GetMaximumBin())
        clone = self.Clone(shallow=True)
        for i in range(self.GetSize()):
            clone.SetBinContent(
                i, clone.GetBinContent(i) + clone.GetBinError(i))
        return clone.GetBinContent(clone.GetMaximumBin())

    def min(self, include_error=False):
        if not include_error:
            return self.GetBinContent(self.GetMinimumBin())
        clone = self.Clone(shallow=True)
        for i in range(self.GetSize()):
            clone.SetBinContent(
                i, clone.GetBinContent(i) - clone.GetBinError(i))
        return clone.GetBinContent(clone.GetMinimumBin())


class _Hist(_HistBase):

    DIM = 1

    def x(self, index=None, overflow=False):
        return self._centers(0, index, overflow=overflow)

    def xerravg(self, index=None, overflow=False):
        return self._erravg(0, index, overflow=overflow)

    def xerrl(self, index=None, overflow=False):
        return self._erravg(0, index, overflow=overflow)

    def xerrh(self, index=None, overflow=False):
        return self._erravg(0, index, overflow=overflow)

    def xerr(self, index=None, overflow=False):
        return self._err(0, index, overflow=overflow)

    def xwidth(self, index=None, overflow=False):
        return self._width(0, index, overflow=overflow)

    def xedgesl(self, index=None, overflow=False):
        return self._edgesl(0, index, overflow=overflow)

    def xedgesh(self, index=None, overflow=False):
        return self._edgesh(0, index, overflow=overflow)

    def xedges(self, index=None, overflow=False):
        return self._edges(0, index, overflow=overflow)

    def yerrh(self, index=None, overflow=False):
        return self.yerravg(index, overflow=overflow)

    def yerrl(self, index=None, overflow=False):
        return self.yerravg(index, overflow=overflow)

    def y(self, index=None, overflow=False):
        if index is None:
            return (self.GetBinContent(i)
                    for i in self.bins_range(axis=0, overflow=overflow))
        index = index % self.nbins(axis=0, overflow=True)
        return self.GetBinContent(index)

    def yerravg(self, index=None, overflow=False):
        if index is None:
            return (self.GetBinError(i)
                    for i in self.bins_range(axis=0, overflow=overflow))
        index = index % self.nbins(axis=0, overflow=True)
        return self.GetBinError(index)

    def yerr(self, index=None, overflow=False):
        if index is None:
            return ((self.yerrl(i), self.yerrh(i))
                    for i in self.bins_range(axis=0, overflow=overflow))
        index = index % self.nbins(axis=0, overflow=True)
        return (self.yerrl(index), self.yerrh(index))

    def expectation(self, startbin=1, endbin=None):
        if endbin is not None and endbin < startbin:
            raise ValueError("``endbin`` should be greated than ``startbin``")
        if endbin is None:
            endbin = self.nbins(0)
        expect = 0.
        norm = 0.
        for index in range(startbin, endbin + 1):
            val = self[index]
            expect += val * self.x(index)
            norm += val
        if norm > 0:
            return expect / norm
        else:
            return (self.xedges(endbin + 1) + self.xedges(startbin)) / 2

    def integral(self, xbin1=None, xbin2=None,
                 width=False, error=False, overflow=False):
        """
        Compute the integral and error over a range of bins
        """
        if xbin1 is None:
            xbin1 = 0 if overflow else 1
        if xbin2 is None:
            xbin2 = -1 if overflow else -2
        nbinsx = self.nbins(axis=0, overflow=True)
        xbin1 %= nbinsx
        xbin2 %= nbinsx
        options = 'width' if width else ''
        if error:
            error = ROOT.Double()
            integral = super(_Hist, self).IntegralAndError(
                xbin1, xbin2, error, options)
            return integral, error
        return super(_Hist, self).Integral(xbin1, xbin2, options)

    def poisson_errors(self):
        """
        Return a TGraphAsymmErrors representation of this histogram where the
        point y errors are Poisson.
        """
        graph = Graph(self.nbins(axis=0), type='asymm')
        graph.SetLineWidth(self.GetLineWidth())
        graph.SetMarkerSize(self.GetMarkerSize())
        chisqr = ROOT.TMath.ChisquareQuantile
        npoints = 0
        for bin in self.bins(overflow=False):
            entries = bin.effective_entries
            if entries <= 0:
                continue
            ey_low = entries - 0.5 * chisqr(0.1586555, 2. * entries)
            ey_high = 0.5 * chisqr(
                1. - 0.1586555, 2. * (entries + 1)) - entries
            ex = bin.x.width / 2.
            graph.SetPoint(npoints, bin.x.center, bin.value)
            graph.SetPointEXlow(npoints, ex)
            graph.SetPointEXhigh(npoints, ex)
            graph.SetPointEYlow(npoints, ey_low)
            graph.SetPointEYhigh(npoints, ey_high)
            npoints += 1
        graph.Set(npoints)
        return graph


class _Hist2D(_HistBase):

    DIM = 2

    def x(self, index=None, overflow=False):
        return self._centers(0, index, overflow=overflow)

    def xerravg(self, index=None, overflow=False):
        return self._erravg(0, index, overflow=overflow)

    def xerrl(self, index=None, overflow=False):
        return self._erravg(0, index, overflow=overflow)

    def xerrh(self, index=None, overflow=False):
        return self._erravg(0, index, overflow=overflow)

    def xerr(self, index=None, overflow=False):
        return self._err(0, index, overflow=overflow)

    def xwidth(self, index=None, overflow=False):
        return self._width(0, index, overflow=overflow)

    def xedgesl(self, index=None, overflow=False):
        return self._edgesl(0, index, overflow=overflow)

    def xedgesh(self, index=None, overflow=False):
        return self._edgesh(0, index, overflow=overflow)

    def xedges(self, index=None, overflow=False):
        return self._edges(0, index, overflow=overflow)

    def y(self, index=None, overflow=False):
        return self._centers(1, index, overflow=overflow)

    def yerravg(self, index=None, overflow=False):
        return self._erravg(1, index, overflow=overflow)

    def yerrl(self, index=None, overflow=False):
        return self._erravg(1, index, overflow=overflow)

    def yerrh(self, index=None, overflow=False):
        return self._erravg(1, index, overflow=overflow)

    def yerr(self, index=None, overflow=False):
        return self._err(1, index, overflow=overflow)

    def ywidth(self, index=None, overflow=False):
        return self._width(1, index, overflow=overflow)

    def yedgesl(self, index=None, overflow=False):
        return self._edgesl(1, index, overflow=overflow)

    def yedgesh(self, index=None, overflow=False):
        return self._edgesh(1, index, overflow=overflow)

    def yedges(self, index=None, overflow=False):
        return self._edges(1, index, overflow=overflow)

    def zerrh(self, index=None, overflow=False):
        return self.zerravg(index, overflow=overflow)

    def zerrl(self, index=None, overflow=False):
        return self.zerravg(index, overflow=overflow)

    def z(self, ix=None, iy=None, overflow=False):
        if ix is None and iy is None:
            return [[self.GetBinContent(ix, iy)
                    for iy in self.bins_range(axis=1, overflow=overflow)]
                    for ix in self.bins_range(axis=0, overflow=overflow)]
        ix = ix % self.nbins(axis=0, overflow=True)
        iy = iy % self.nbins(axis=1, overflow=True)
        return self.GetBinContent(ix, iy)

    def zerravg(self, ix=None, iy=None, overflow=False):
        if ix is None and iy is None:
            return [[self.GetBinError(ix, iy)
                    for iy in self.bins_range(axis=1, overflow=overflow)]
                    for ix in self.bins_range(axis=0, overflow=overflow)]
        ix = ix % self.nbins(axis=0, overflow=True)
        iy = iy % self.nbins(axis=1, overflow=True)
        return self.GetBinError(ix, iy)

    def zerr(self, ix=None, iy=None, overflow=False):
        if ix is None and iy is None:
            return [[(self.GetBinError(ix, iy), self.GetBinError(ix, iy))
                    for iy in self.bins_range(axis=1, overflow=overflow)]
                    for ix in self.bins_range(axis=0, overflow=overflow)]
        ix = ix % self.nbins(axis=0, overflow=True)
        iy = iy % self.nbins(axis=1, overflow=True)
        return (self.GetBinError(ix, iy),
                self.GetBinError(ix, iy))

    def ravel(self, name=None):
        """
        Convert 2D histogram into 1D histogram with the y-axis repeated along
        the x-axis, similar to NumPy's ravel().
        """
        nbinsx = self.nbins(0)
        nbinsy = self.nbins(1)
        left_edge = self.xedgesl(1)
        right_edge = self.xedgesh(nbinsx)
        out = Hist(nbinsx * nbinsy,
                   left_edge, nbinsy * (right_edge - left_edge) + left_edge,
                   type=self.TYPE,
                   name=name,
                   title=self.title,
                   **self.decorators)
        for i, bin in enumerate(self.bins(overflow=False)):
            out.SetBinContent(i + 1, bin.value)
            out.SetBinError(i + 1, bin.error)
        return out

    def integral(self,
                 xbin1=None, xbin2=None,
                 ybin1=None, ybin2=None,
                 width=False,
                 error=False,
                 overflow=False):
        """
        Compute the integral and error over a range of bins
        """
        if xbin1 is None:
            xbin1 = 0 if overflow else 1
        if xbin2 is None:
            xbin2 = -1 if overflow else -2
        if ybin1 is None:
            ybin1 = 0 if overflow else 1
        if ybin2 is None:
            ybin2 = -1 if overflow else -2
        nbinsx = self.nbins(axis=0, overflow=True)
        xbin1 %= nbinsx
        xbin2 %= nbinsx
        nbinsy = self.nbins(axis=1, overflow=True)
        ybin1 %= nbinsy
        ybin2 %= nbinsy
        options = 'width' if width else ''
        if error:
            error = ROOT.Double()
            integral = super(_Hist2D, self).IntegralAndError(
                xbin1, xbin2, ybin1, ybin2, error, options)
            return integral, error
        return super(_Hist2D, self).Integral(
            xbin1, xbin2, ybin1, ybin2, options)


class _Hist3D(_HistBase):

    DIM = 3

    def x(self, index=None, overflow=False):
        return self._centers(0, index, overflow=overflow)

    def xerravg(self, index=None, overflow=False):
        return self._erravg(0, index, overflow=overflow)

    def xerrl(self, index=None, overflow=False):
        return self._erravg(0, index, overflow=overflow)

    def xerrh(self, index=None, overflow=False):
        return self._erravg(0, index, overflow=overflow)

    def xerr(self, index=None, overflow=False):
        return self._err(0, index, overflow=overflow)

    def xwidth(self, index=None, overflow=False):
        return self._width(0, index, overflow=overflow)

    def xedgesl(self, index=None, overflow=False):
        return self._edgesl(0, index, overflow=overflow)

    def xedgesh(self, index=None, overflow=False):
        return self._edgesh(0, index, overflow=overflow)

    def xedges(self, index=None, overflow=False):
        return self._edges(0, index, overflow=overflow)

    def y(self, index=None, overflow=False):
        return self._centers(1, index, overflow=overflow)

    def yerravg(self, index=None, overflow=False):
        return self._erravg(1, index, overflow=overflow)

    def yerrl(self, index=None, overflow=False):
        return self._erravg(1, index, overflow=overflow)

    def yerrh(self, index=None, overflow=False):
        return self._erravg(1, index, overflow=overflow)

    def yerr(self, index=None, overflow=False):
        return self._err(1, index, overflow=overflow)

    def ywidth(self, index=None, overflow=False):
        return self._width(1, index, overflow=overflow)

    def yedgesl(self, index=None, overflow=False):
        return self._edgesl(1, index, overflow=overflow)

    def yedgesh(self, index=None, overflow=False):
        return self._edgesh(1, index, overflow=overflow)

    def yedges(self, index=None, overflow=False):
        return self._edges(1, index, overflow=overflow)

    def z(self, index=None, overflow=False):
        return self._centers(2, index, overflow=overflow)

    def zerravg(self, index=None, overflow=False):
        return self._erravg(2, index, overflow=overflow)

    def zerrl(self, index=None, overflow=False):
        return self._erravg(2, index, overflow=overflow)

    def zerrh(self, index=None, overflow=False):
        return self._erravg(2, index, overflow=overflow)

    def zerr(self, index=None, overflow=False):
        return self._err(2, index, overflow=overflow)

    def zwidth(self, index=None, overflow=False):
        return self._width(2, index, overflow=overflow)

    def zedgesl(self, index=None, overflow=False):
        return self._edgesl(2, index, overflow=overflow)

    def zedgesh(self, index=None, overflow=False):
        return self._edgesh(2, index, overflow=overflow)

    def zedges(self, index=None, overflow=False):
        return self._edges(2, index, overflow=overflow)

    def werrh(self, index=None, overflow=False):
        return self.werravg(index, overflow=overflow)

    def werrl(self, index=None, overflow=False):
        return self.werravg(index, overflow=overflow)

    def w(self, ix=None, iy=None, iz=None, overflow=False):
        if ix is None and iy is None and iz is None:
            return [[[self.GetBinContent(ix, iy, iz)
                    for iz in self.bins_range(axis=2, overflow=overflow)]
                    for iy in self.bins_range(axis=1, overflow=overflow)]
                    for ix in self.bins_range(axis=0, overflow=overflow)]
        ix = ix % self.nbins(axis=0, overflow=True)
        iy = iy % self.nbins(axis=1, overflow=True)
        iz = iz % self.nbins(axis=2, overflow=True)
        return self.GetBinContent(ix, iy, iz)

    def werravg(self, ix=None, iy=None, iz=None, overflow=False):
        if ix is None and iy is None and iz is None:
            return [[[self.GetBinError(ix, iy, iz)
                    for iz in self.bins_range(axis=2, overflow=overflow)]
                    for iy in self.bins_range(axis=1, overflow=overflow)]
                    for ix in self.bins_range(axis=0, overflow=overflow)]
        ix = ix % self.nbins(axis=0, overflow=True)
        iy = iy % self.nbins(axis=1, overflow=True)
        iz = iz % self.nbins(axis=2, overflow=True)
        return self.GetBinError(ix, iy, iz)

    def werr(self, ix=None, iy=None, iz=None, overflow=False):
        if ix is None and iy is None and iz is None:
            return [[[
                (self.GetBinError(ix, iy, iz), self.GetBinError(ix, iy, iz))
                for iz in self.bins_range(axis=2, overflow=overflow)]
                for iy in self.bins_range(axis=1, overflow=overflow)]
                for ix in self.bins_range(axis=0, overflow=overflow)]
        ix = ix % self.nbins(axis=0, overflow=True)
        iy = iy % self.nbins(axis=1, overflow=True)
        iz = iz % self.nbins(axis=2, overflow=True)
        return (self.GetBinError(ix, iy, iz),
                self.GetBinError(ix, iy, iz))

    def integral(self,
                 xbin1=1, xbin2=-2,
                 ybin1=1, ybin2=-2,
                 zbin1=1, zbin2=-2,
                 width=False,
                 error=False,
                 overflow=False):
        """
        Compute the integral and error over a range of bins
        """
        if xbin1 is None:
            xbin1 = 0 if overflow else 1
        if xbin2 is None:
            xbin2 = -1 if overflow else -2
        if ybin1 is None:
            ybin1 = 0 if overflow else 1
        if ybin2 is None:
            ybin2 = -1 if overflow else -2
        if zbin1 is None:
            zbin1 = 0 if overflow else 1
        if zbin2 is None:
            zbin2 = -1 if overflow else -2
        nbinsx = self.nbins(axis=0, overflow=True)
        xbin1 %= nbinsx
        xbin2 %= nbinsx
        nbinsy = self.nbins(axis=1, overflow=True)
        ybin1 %= nbinsy
        ybin2 %= nbinsy
        nbinsz = self.nbins(axis=2, overflow=True)
        zbin1 %= nbinsz
        zbin2 %= nbinsz
        options = 'width' if width else ''
        if error:
            error = ROOT.Double()
            integral = super(_Hist3D, self).IntegralAndError(
                xbin1, xbin2, ybin1, ybin2, zbin1, zbin2, error, options)
            return integral, error
        return super(_Hist3D, self).Integral(
            xbin1, xbin2, ybin1, ybin2, zbin1, zbin2, options)


def _Hist_class(type='F'):
    type = type.upper()
    if type not in _HistBase.TYPES:
        raise TypeError(
            "No histogram available with bin type {0}".format(type))
    rootclass = _HistBase.TYPES[type][0]

    class Hist(_Hist, rootclass):
        _ROOT = rootclass
        TYPE = type

        def __init__(self, *args, **kwargs):
            params = self._parse_args(args)
            name = kwargs.pop('name', None)
            title = kwargs.pop('title', None)
            if params[0]['bins'] is None:
                super(Hist, self).__init__(
                    params[0]['nbins'], params[0]['low'], params[0]['high'],
                    name=name, title=title)
            else:
                super(Hist, self).__init__(
                    params[0]['nbins'], array('d', params[0]['bins']),
                    name=name, title=title)
            self._post_init(**kwargs)

    return Hist


def _Hist2D_class(type='F'):
    type = type.upper()
    if type not in _HistBase.TYPES:
        raise TypeError(
            "No histogram available with bin type {0}".format(type))
    rootclass = _HistBase.TYPES[type][1]

    class Hist2D(_Hist2D, rootclass):
        _ROOT = rootclass
        TYPE = type

        def __init__(self, *args, **kwargs):
            params = self._parse_args(args)
            name = kwargs.pop('name', None)
            title = kwargs.pop('title', None)
            if params[0]['bins'] is None and params[1]['bins'] is None:
                super(Hist2D, self).__init__(
                    params[0]['nbins'], params[0]['low'], params[0]['high'],
                    params[1]['nbins'], params[1]['low'], params[1]['high'],
                    name=name, title=title)
            elif params[0]['bins'] is None and params[1]['bins'] is not None:
                super(Hist2D, self).__init__(
                    params[0]['nbins'], params[0]['low'], params[0]['high'],
                    params[1]['nbins'], array('d', params[1]['bins']),
                    name=name, title=title)
            elif params[0]['bins'] is not None and params[1]['bins'] is None:
                super(Hist2D, self).__init__(
                    params[0]['nbins'], array('d', params[0]['bins']),
                    params[1]['nbins'], params[1]['low'], params[1]['high'],
                    name=name, title=title)
            else:
                super(Hist2D, self).__init__(
                    params[0]['nbins'], array('d', params[0]['bins']),
                    params[1]['nbins'], array('d', params[1]['bins']),
                    name=name, title=title)
            self._post_init(**kwargs)

    return Hist2D


def _Hist3D_class(type='F'):
    type = type.upper()
    if type not in _HistBase.TYPES:
        raise TypeError(
            "No histogram available with bin type {0}".format(type))
    rootclass = _HistBase.TYPES[type][2]

    class Hist3D(_Hist3D, rootclass):
        _ROOT = rootclass
        TYPE = type

        def __init__(self, *args, **kwargs):
            params = self._parse_args(args)
            name = kwargs.pop('name', None)
            title = kwargs.pop('title', None)
            # ROOT is missing constructors for TH3...
            if (params[0]['bins'] is None and
                    params[1]['bins'] is None and
                    params[2]['bins'] is None):
                super(Hist3D, self).__init__(
                    params[0]['nbins'], params[0]['low'], params[0]['high'],
                    params[1]['nbins'], params[1]['low'], params[1]['high'],
                    params[2]['nbins'], params[2]['low'], params[2]['high'],
                    name=name, title=title)
            else:
                if params[0]['bins'] is None:
                    step = ((params[0]['high'] - params[0]['low'])
                            / float(params[0]['nbins']))
                    params[0]['bins'] = [
                        params[0]['low'] + n * step
                        for n in range(params[0]['nbins'] + 1)]
                if params[1]['bins'] is None:
                    step = ((params[1]['high'] - params[1]['low'])
                            / float(params[1]['nbins']))
                    params[1]['bins'] = [
                        params[1]['low'] + n * step
                        for n in range(params[1]['nbins'] + 1)]
                if params[2]['bins'] is None:
                    step = ((params[2]['high'] - params[2]['low'])
                            / float(params[2]['nbins']))
                    params[2]['bins'] = [
                        params[2]['low'] + n * step
                        for n in range(params[2]['nbins'] + 1)]
                super(Hist3D, self).__init__(
                    params[0]['nbins'], array('d', params[0]['bins']),
                    params[1]['nbins'], array('d', params[1]['bins']),
                    params[2]['nbins'], array('d', params[2]['bins']),
                    name=name, title=title)
            self._post_init(**kwargs)

    return Hist3D


_HIST_CLASSES_1D = {}
_HIST_CLASSES_2D = {}
_HIST_CLASSES_3D = {}

for bintype in _HistBase.TYPES.keys():
    cls = _Hist_class(type=bintype)
    snake_case_methods(cls)
    _HIST_CLASSES_1D[bintype] = cls

    cls = _Hist2D_class(type=bintype)
    snake_case_methods(cls)
    _HIST_CLASSES_2D[bintype] = cls

    cls = _Hist3D_class(type=bintype)
    snake_case_methods(cls)
    _HIST_CLASSES_3D[bintype] = cls


class Hist(_Hist, QROOT.TH1):
    """
    Returns a 1-dimensional Hist object which inherits from the associated
    ROOT.TH1* class (where * is C, S, I, F, or D depending on the type
    keyword argument)
    """
    _ROOT = QROOT.TH1

    @classmethod
    def dynamic_cls(cls, type='F'):
        return _HIST_CLASSES_1D[type]

    def __new__(cls, *args, **kwargs):
        if len(args) == 1:
            other = args[0]
            kwargs.setdefault('type', 'F')
            if isinstance(other, HistView):
                obj = Hist(other.xedges, **kwargs)
                obj.fill_view(other.hist[:])
                obj.entries = other.hist.entries
                return obj
            elif isinstance(other, _Hist):
                obj = other.empty_clone(**kwargs)
                obj[:] = other[:]
                obj.entries = other.entries
                return obj
            elif isinstance(other, _Graph1DBase):
                # attempt to convert graph to histogram
                if len(other) == 0:
                    raise ValueError("cannot construct a histogram "
                                     "from an empty graph")
                edges = [other.x(0) - other.xerrl(0)] # first edge
                values = []
                errors = []
                for ipoint in range(len(other)):
                    edges.append(other.x(ipoint) + other.xerrh(ipoint))
                    values.append(other.y(ipoint))
                    errors.append(max(abs(other.yerrh(ipoint)),
                                      abs(other.yerrl(ipoint))))
                obj = Hist(edges, **kwargs)
                for idx, (y, yerr) in enumerate(zip(values, errors)):
                    obj[idx + 1] = (y, yerr)
                return obj

        type = kwargs.pop('type', 'F').upper()
        return cls.dynamic_cls(type)(*args, **kwargs)


# alias Hist1D -> Hist
Hist1D = Hist


class Hist2D(_Hist2D, QROOT.TH2):
    """
    Returns a 2-dimensional Hist object which inherits from the associated
    ROOT.TH1* class (where * is C, S, I, F, or D depending on the type
    keyword argument)
    """
    _ROOT = QROOT.TH2

    @classmethod
    def dynamic_cls(cls, type='F'):
        return _HIST_CLASSES_2D[type]

    def __new__(cls, *args, **kwargs):
        if len(args) == 1:
            other = args[0]
            kwargs.setdefault('type', 'F')
            if isinstance(other, Hist2DView):
                obj = Hist2D(other.xedges, other.yedges, **kwargs)
                obj.fill_view(other.hist[:,:])
                obj.entries = other.hist.entries
                return obj
            elif isinstance(other, _Hist2D):
                obj = other.empty_clone(**kwargs)
                obj[:] = other[:]
                obj.entries = other.entries
                return obj
        type = kwargs.pop('type', 'F').upper()
        return cls.dynamic_cls(type)(*args, **kwargs)


class Hist3D(_Hist3D, QROOT.TH3):
    """
    Returns a 3-dimensional Hist object which inherits from the associated
    ROOT.TH1* class (where * is C, S, I, F, or D depending on the type
    keyword argument)
    """
    _ROOT = QROOT.TH3

    @classmethod
    def dynamic_cls(cls, type='F'):
        return _HIST_CLASSES_3D[type]

    def __new__(cls, *args, **kwargs):
        if len(args) == 1:
            other = args[0]
            kwargs.setdefault('type', 'F')
            if isinstance(other, Hist3DView):
                obj = Hist3D(other.xedges, other.yedges, other.zedges, **kwargs)
                obj.fill_view(other.hist[:,:,:])
                obj.entries = other.hist.entries
                return obj
            elif isinstance(other, _Hist3D):
                obj = other.empty_clone(**kwargs)
                obj[:] = other[:]
                obj.entries = other.entries
                return obj
        type = kwargs.pop('type', 'F').upper()
        return cls.dynamic_cls(type)(
            *args, **kwargs)


class HistStack(Plottable, NamedObject, QROOT.THStack):

    _ROOT = QROOT.THStack

    def __init__(self, hists=None, name=None, title=None,
                 stacked=True, **kwargs):
        super(HistStack, self).__init__(name=name, title=title)
        self._post_init(hists=hists, stacked=stacked, **kwargs)

    def _post_init(self, hists=None, stacked=True, **kwargs):
        super(HistStack, self)._post_init(**kwargs)
        self.hists = []
        self.dim = 1
        current_hists = super(HistStack, self).GetHists()
        if current_hists:
            for i, hist in enumerate(current_hists):
                hist = asrootpy(hist)
                if i == 0:
                    self.dim = dim(hist)
                elif dim(hist) != self.dim:
                    raise TypeError(
                        "Dimensions of the contained histograms are not equal")
                self.hists.append(hist)
        self.stacked = stacked
        if stacked:
            # histogram binning must be identical
            self.sum = sum(self.hists) if self.hists else None
        if hists:
            for h in hists:
                self.Add(h)

    def __dim__(self):
        return self.dim

    def GetHists(self):
        return [hist for hist in self.hists]

    def Add(self, hist):
        if isinstance(hist, _Hist) or isinstance(hist, _Hist2D):
            if not self:
                self.dim = dim(hist)
                if self.stacked:
                    self.sum = hist.Clone()
            elif dim(self) != dim(hist):
                raise TypeError(
                    "Dimension of histogram does not match dimension "
                    "of already contained histograms")
            elif self.stacked:
                self.sum += hist
            self.hists.append(hist)
            super(HistStack, self).Add(hist, hist.drawstyle)
        else:
            raise TypeError(
                "Only 1D and 2D histograms are supported")

    def __add__(self, other):
        if not isinstance(other, HistStack):
            raise TypeError(
                "Addition not supported for HistStack and {0}".format(
                    other.__class__.__name__))
        clone = HistStack()
        for hist in self:
            clone.Add(hist)
        for hist in other:
            clone.Add(hist)
        return clone

    def __iadd__(self, other):
        if not isinstance(other, HistStack):
            raise TypeError(
                "Addition not supported for HistStack and {0}".format(
                    other.__class__.__name__))
        for hist in other:
            self.Add(hist)
        return self

    def __len__(self):
        return len(self.GetHists())

    def __getitem__(self, index):
        return self.GetHists()[index]

    def __iter__(self):
        for hist in self.hists:
            yield hist

    def __nonzero__(self):
        return len(self) != 0

    __bool__ = __nonzero__

    def __cmp__(self, other):
        diff = self.max() - other.max()
        if diff > 0:
            return 1
        if diff < 0:
            return -1
        return 0

    def Scale(self, value):
        for hist in self:
            hist.Scale(value)

    def Integral(self, start=None, end=None):
        integral = 0
        if start is not None and end is not None:
            for hist in self:
                integral += hist.Integral(start, end)
        else:
            for hist in self:
                integral += hist.Integral()
        return integral

    def lowerbound(self, axis=0):
        if not self:
            return None  # negative infinity
        return min(hist.lowerbound(axis=axis) for hist in self)

    def upperbound(self, axis=0):
        if not self:
            return ()  # positive infinity
        return max(hist.upperbound(axis=axis) for hist in self)

    def max(self, include_error=False):
        if not self:
            return 0
        if self.stacked:
            return self.sum.max(include_error=include_error)
        return max([hist.max(include_error=include_error)
                    for hist in self.hists])

    def min(self, include_error=False):
        if not self:
            return 0
        if self.stacked:
            return self.sum.min(include_error=include_error)
        return min([hist.min(include_error=include_error)
                    for hist in self.hists])

    def Clone(self, newName=None):
        clone = HistStack(name=newName,
                          title=self.GetTitle(),
                          **self.decorators)
        for hist in self:
            clone.Add(hist.Clone())
        return clone

    def GetHistogram(self):
        return asrootpy(super(HistStack, self).GetHistogram())

    def GetZaxis(self):
        # ROOT is missing this method...
        return self.GetHistogram().zaxis


@snake_case_methods
class Efficiency(Plottable, NamelessConstructorObject, QROOT.TEfficiency):
    _ROOT = QROOT.TEfficiency

    def __init__(self, passed, total, name=None, title=None, **kwargs):
        super(Efficiency, self).__init__(passed, total, name=name, title=title)
        self._post_init(**kwargs)

    @property
    def passed(self):
        return asrootpy(self.GetPassedHistogram())

    @passed.setter
    def passed(self, hist):
        self.SetPassedHistogram(hist, 'f')

    @property
    def total(self):
        return asrootpy(self.GetTotalHistogram())

    @total.setter
    def total(self, hist):
        self.SetTotalHistogram(hist, 'f')

    def __len__(self):
        return len(self.total)

    def __getitem__(self, idx):
        return self.GetEfficiency(idx)

    def __add__(self, other):
        copy = self.Clone()
        copy.Add(other)
        return copy

    def __iadd__(self, other):
        super(Efficiency, self).Add(self, other)
        return self

    def __iter__(self):
        for idx in range(len(self)):
            yield self.GetEfficiency(idx)

    def efficiencies(self, overflow=False):
        if overflow:
            start = 0
            end = len(self)
        else:
            start = 1
            end = len(self) - 1
        for idx in range(start, end):
            yield self.GetEfficiency(idx)

    def errors(self, overflow=False):
        if overflow:
            start = 0
            end = len(self)
        else:
            start = 1
            end = len(self) - 1
        for idx in range(start, end):
            yield (
                self.GetEfficiencyErrorLow(idx),
                self.GetEfficiencyErrorUp(idx))

    def overall_efficiency(self, overflow=False):
        if self.total.Integral() == 0:
            return 0

        nbins = self.passed.nbins()
        if overflow:
            bins_to_merge = (0, nbins+1)
        else:
            bins_to_merge = (1, nbins)

        hpass = self.passed.merge_bins([bins_to_merge])
        htot = self.total.merge_bins([bins_to_merge])
        tot_eff = Efficiency(hpass, htot)
        return (tot_eff.GetEfficiency(1),
                tot_eff.GetEfficiencyErrorLow(1),
                tot_eff.GetEfficiencyErrorUp(1))

    @property
    def graph(self):
        """ Create and return the graph for a 1D TEfficiency """
        return asrootpy(self.CreateGraph())

    @property
    def painted_graph(self):
        """
        Returns the painted graph for a 1D TEfficiency, or if it isn't
        available, generates one on an `invisible_canvas`.
        """
        if not self.GetPaintedGraph():
            with invisible_canvas():
                self.Draw()
        assert self.GetPaintedGraph(), (
            "Failed to create TEfficiency::GetPaintedGraph")
        the_graph = asrootpy(self.GetPaintedGraph())
        # Ensure it has the same style as this one.
        the_graph.decorate(**self.decorators)
        return the_graph

    @property
    def histogram(self):
        """ Create and return the histogram for a 2D TEfficiency """
        return asrootpy(self.CreateHistogram())

    @property
    def painted_histogram(self):
        """
        Returns the painted histogram for a 2D TEfficiency, or if it isn't
        available, generates one on an `invisible_canvas`.
        """
        if not self.GetPaintedHistogram():
            with invisible_canvas():
                self.Draw()
        assert self.GetPaintedHistogram(), (
            "Failed to create TEfficiency::GetPaintedHistogram")
        the_hist = asrootpy(self.GetPaintedHistogram())
        # Ensure it has the same style as this one.
        the_hist.decorate(**self.decorators)
        return the_hist


def histogram(data, *args, **kwargs):
    """
    Create and fill a one-dimensional histogram.

    The same arguments as the ``Hist`` class are expected.
    If the number of bins and the ranges are not specified they are
    automatically deduced with the ``autobinning`` function using the method
    specified by the ``binning`` argument. Only one-dimensional histogramming
    is supported.
    """
    from .autobinning import autobinning
    dim = kwargs.pop('dim', 1)
    if dim != 1:
        raise NotImplementedError
    if 'binning' in kwargs:
        args = autobinning(data, kwargs['binning'])
        del kwargs['binning']
    histo = Hist(*args, **kwargs)
    for d in data:
        histo.Fill(d)
    return list(histo.xedgesl()), histo
