# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
from array import array

from itertools import product

import ROOT

from .. import asrootpy, QROOT, log; log = log[__name__],
from ..core import NamedObject, isbasictype
from ..decorators import snake_case_methods
from .core import Plottable, dim
from ..context import invisible_canvas
from ..objectproxy import ObjectProxy
from .graph import Graph


class DomainError(Exception):
    pass

class BinProxy(object):

    def __init__(self, hist, idx):
        self.hist, self.idx = hist, idx

    @property
    def overflow(self):
        """
        Returns true if this BinProxy is for an overflow bin
        """
        indices = self.hist.xyz(self.idx)
        for i in xrange(1, self.hist.GetDimension()+1):
            if indices[i-1] == 0 or indices[i-1] == self.hist.nbins(i) + 1:
                return True

    @property
    def x(self):
        return self.hist.axis_bininfo(1, self.axis_idx(0))
    @property
    def y(self):
        return self.hist.axis_bininfo(2, self.axis_idx(1))
    @property
    def z(self):
        return self.hist.axis_bininfo(3, self.axis_idx(2))

    def axis_idx(self, axi):
        """
        Index for this bin along the axi'th axis
        """
        return self.hist.xyz(self.idx)[axi]

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

    def __imul__(self, v):
        self.value *= v
        self.error *= v

class _HistBase(Plottable, NamedObject):

    def xyz(self, i):
        x, y, z = ROOT.Long(0), ROOT.Long(0), ROOT.Long(0)
        self.GetBinXYZ(i, x, y, z)
        return x, y, z

    def axis_bininfo(self, axi, i):
        class bi:
            ax = self.axis(axi)
            lo = ax.GetBinLowEdge(i)
            center = ax.GetBinCenter(i)
            up = ax.GetBinUpEdge(i)
            width = ax.GetBinWidth(i)
        return bi

    def bins(self, overflow=False):
        for i in xrange(self.GetSize()):
            bproxy = BinProxy(self, i)
            if not overflow and bproxy.overflow:
                continue
            yield bproxy

    TYPES = dict((c, [getattr(QROOT, "TH{0}{1}".format(d, c))
                      for d in (1, 2, 3)])
                 for c in "CSIFD")

    def _parse_args(self, args, ignore_extras=False):

        params = [{'bins': None,
                   'nbins': None,
                   'low': None,
                   'high': None} for _ in xrange(dim(self))]

        for param in params:
            if len(args) == 0:
                raise TypeError("Did not receive expected number of arguments")
            if hasattr(args[0], '__iter__'):
                if list(sorted(args[0])) != list(args[0]):
                    raise ValueError(
                        "Bin edges must be sorted in ascending order")
                if len(set(args[0])) != len(args[0]):
                    raise ValueError("Bin edges must not be repeated")
                param['bins'] = args[0]
                param['nbins'] = len(args[0]) - 1
                args = args[1:]
            elif len(args) >= 3:
                nbins = args[0]
                if type(nbins) is not int:
                    raise TypeError(
                        "Type of first argument (got %s %s) must be an int" %
                        (type(nbins), nbins))
                low = args[1]
                if not isbasictype(low):
                    raise TypeError(
                        "Type of second argument must be int, float, or long")
                high = args[2]
                if not isbasictype(high):
                    raise TypeError(
                        "Type of third argument must be int, float, or long")
                param['nbins'] = nbins
                param['low'] = low
                param['high'] = high
                if low >= high:
                    raise ValueError(
                        "Upper bound (you gave %f) must be greater than lower "
                        "bound (you gave %f)" % (float(low), float(high)))
                args = args[3:]
            else:
                raise TypeError(
                    "Did not receive expected number of arguments")

        if ignore_extras:
            # used by Profile where range of profiled axis may be specified
            return params, args

        if len(args) != 0:
            raise TypeError(
                "Did not receive expected number of arguments")

        return params

    @classmethod
    def divide(cls, h1, h2, c1=1., c2=1., option=''):

        ratio = h1.Clone()
        ROOT.TH1.Divide(ratio, h1, h2, c1, c2, option)
        return ratio

    def Fill(self, *args):

        bin = self.ROOT_base.Fill(self, *args)
        if bin > 0:
            return bin - 1
        return bin

    def nbins(self, axis=1):

        if axis == 1:
            return self.GetNbinsX()
        elif axis == 2:
            return self.GetNbinsY()
        elif axis == 3:
            return self.GetNbinsZ()
        else:
            raise ValueError("%s is not a valid axis index!" % axis)

    @property
    def axes(self):
        return [self.axis(i) for i in xrange(1, self.GetDimension()+1)]

    def axis(self, axis=1):

        if axis == 1:
            return self.GetXaxis()
        elif axis == 2:
            return self.GetYaxis()
        elif axis == 3:
            return self.GetZaxis()
        else:
            raise ValueError("%s is not a valid axis index!" % axis)

    def underflow(self, axis=1):
        """
        Return the underflow for the given axis.

        Depending on the dimension of the histogram, may return an array.
        """
        if axis not in [1, 2, 3]:
            raise ValueError("%s is not a valid axis index!" % axis)
        if self.DIM == 1:
            return self.GetBinContent(0)
        elif self.DIM == 2:
            return [self.GetBinContent(*[i].insert(axis - 1, 0))
                    for i in xrange(self.nbins((axis + 1) % 2))]
        elif self.DIM == 3:
            axis2, axis3 = [1, 2, 3].remove(axis)
            return [[self.GetBinContent(*[i, j].insert(axis - 1, 0))
                     for i in xrange(self.nbins(axis2))]
                    for j in xrange(self.nbins(axis3))]

    def overflow(self, axis=1):
        """
        Return the overflow for the given axis.

        Depending on the dimension of the histogram, may return an array.
        """
        if axis not in [1, 2, 3]:
            raise ValueError("%s is not a valid axis index!" % axis)
        if self.DIM == 1:
            return self.GetBinContent(self.nbins(1) + 1)
        elif self.DIM == 2:
            axis2 = [1, 2].remove(axis)
            return [self.GetBinContent(*[i].insert(axis - 1, self.nbins(axis)))
                    for i in xrange(self.nbins(axis2))]
        elif self.DIM == 3:
            axis2, axis3 = [1, 2, 3].remove(axis)
            return [[self.GetBinContent(
                     *[i, j].insert(axis - 1, self.nbins(axis)))
                     for i in xrange(self.nbins(axis2))]
                     for j in xrange(self.nbins(axis3))]

    def lowerbound(self, axis=1):

        if axis == 1:
            return self.xedges(0)
        if axis == 2:
            return self.yedges(0)
        if axis == 3:
            return self.zedges(0)
        return ValueError("axis must be 1, 2, or 3")

    def upperbound(self, axis=1):

        if axis == 1:
            return self.xedges(-1)
        if axis == 2:
            return self.yedges(-1)
        if axis == 3:
            return self.zedges(-1)
        return ValueError("axis must be 1, 2, or 3")

    def _centers(self, axis, index=None):

        if index is None:
            return (self._centers(axis, i) for i in xrange(self.nbins(axis)))
        index = index % self.nbins(axis)
        return (self._edgesl(axis, index) + self._edgesh(axis, index)) / 2

    def _edgesl(self, axis, index=None):

        if index is None:
            return (self._edgesl(axis, i) for i in xrange(self.nbins(axis)))
        index = index % self.nbins(axis)
        return self.axis(axis).GetBinLowEdge(index + 1)

    def _edgesh(self, axis, index=None):

        if index is None:
            return (self._edgesh(axis, i) for i in xrange(self.nbins(axis)))
        index = index % self.nbins(axis)
        return self.axis(axis).GetBinUpEdge(index + 1)

    def _edges(self, axis, index=None, overflow=False):

        nbins = self.nbins(axis)
        if index is None:
            def temp_generator():
                if overflow:
                    yield float("-inf")
                for index in xrange(nbins):
                    yield self._edgesl(axis, index)
                yield self._edgesh(axis, index)
                if overflow:
                    yield float("+inf")
            return temp_generator()
        index = index % (nbins + 1)
        if index == nbins:
            return self._edgesh(axis, -1)
        return self._edgesl(axis, index)

    def _width(self, axis, index=None):

        if index is None:
            return (self._width(axis, i) for i in xrange(self.nbins(axis)))
        index = index % self.nbins(axis)
        return self._edgesh(axis, index) - self._edgesl(axis, index)

    def _erravg(self, axis, index=None):

        if index is None:
            return (self._erravg(axis, i) for i in xrange(self.nbins(axis)))
        index = index % self.nbins(axis)
        return self._width(axis, index) / 2

    def _err(self, axis, index=None):

        if index is None:
            return ((self._erravg(axis, i), self._erravg(axis, i))
                    for i in xrange(self.nbins(axis)))
        index = index % self.nbins(axis)
        return (self._erravg(axis, index), self._erravg(axis, index))

    def __add__(self, other):

        copy = self.Clone()
        copy += other
        return copy

    def __iadd__(self, other):

        if isbasictype(other):
            if not isinstance(self, _Hist):
                raise ValueError(
                    "A multidimensional histogram must be filled with a tuple")
            self.Fill(other)
        elif type(other) in [list, tuple]:
            if dim(self) not in [len(other), len(other) - 1]:
                raise ValueError(
                    "Dimension of %s does not match dimension "
                    "of histogram (with optional weight as last element)" %
                    str(other))
            self.Fill(*other)
        else:
            self.Add(other)
        return self

    def __sub__(self, other):

        copy = self.Clone()
        copy -= other
        return copy

    def __isub__(self, other):

        if isbasictype(other):
            if not isinstance(self, _Hist):
                raise ValueError(
                    "A multidimensional histogram must be filled with a tuple")
            self.Fill(other, -1)
        elif type(other) in [list, tuple]:
            if len(other) == dim(self):
                self.Fill(*(other + (-1, )))
            elif len(other) == dim(self) + 1:
                # negate last element
                self.Fill(*(other[:-1] + (-1 * other[-1], )))
            else:
                raise ValueError(
                    "Dimension of %s does not match dimension "
                    "of histogram (with optional weight as last element)" %
                    str(other))
        else:
            self.Add(other, -1.)
        return self

    def __mul__(self, other):

        copy = self.Clone()
        copy *= other
        return copy

    def __imul__(self, other):

        if isbasictype(other):
            self.Scale(other)
            return self
        self.Multiply(other)
        return self

    def __div__(self, other):

        copy = self.Clone()
        copy /= other
        return copy

    def __idiv__(self, other):

        if isbasictype(other):
            if other == 0:
                raise ZeroDivisionError()
            self.Scale(1. / other)
            return self
        self.Divide(other)
        return self

    def __radd__(self, other):

        if other == 0:
            return self.Clone()
        raise TypeError("unsupported operand type(s) for +: '%s' and '%s'" %
                (other.__class__.__name__, self.__class__.__name__))

    def __rsub__(self, other):

        if other == 0:
            return self.Clone()
        raise TypeError("unsupported operand type(s) for -: '%s' and '%s'" %
                (other.__class__.__name__, self.__class__.__name__))

    def __len__(self):

        return self.nbins(axis=1)

    def _range_check(self, index, axis=1):

        if not 0 <= index < self.nbins(axis=axis):
            raise IndexError("bin index %i along axis %d is out of range" %
                             (index, axis))

    def __iter__(self):

        return iter(self._content())

    def __cmp__(self, other):

        diff = self.maximum() - other.maximum()
        if diff > 0:
            return 1
        if diff < 0:
            return -1
        return 0

    def errors(self):

        return iter(self._error_content())

    def asarray(self, typecode='f'):

        return array(typecode, self._content())

    def fill_array(self, array, weights=None):
        """
        Fill this histogram with a NumPy array
        """
        try:
            from root_numpy import fill_array
        except ImportError:
            log.critical("root_numpy is needed for Hist*.fill_array. Is it "
                         "installed and importable?")
            raise
        fill_array(self, array, weights=weights)

    def quantiles(self, quantiles):
        qs = array('d', quantiles)
        output = array('d', [0.]*len(quantiles))
        self.GetQuantiles(len(quantiles), output, qs)
        return list(output)


class _Hist(_HistBase):

    DIM = 1

    def x(self, index=None):

        return self._centers(1, index)

    def xerravg(self, index=None):

        return self._erravg(1, index)

    def xerrl(self, index=None):

        return self._erravg(1, index)

    def xerrh(self, index=None):

        return self._erravg(1, index)

    def xerr(self, index=None):

        return self._err(1, index)

    def xwidth(self, index=None):

        return self._width(1, index)

    def xedgesl(self, index=None):

        return self._edgesl(1, index)

    def xedgesh(self, index=None):

        return self._edgesh(1, index)

    def xedges(self, index=None):

        return self._edges(1, index)

    def yerrh(self, index=None):

        return self.yerravg(index)

    def yerrl(self, index=None):

        return self.yerravg(index)

    def y(self, index=None):

        if index is None:
            return (self.y(i) for i in xrange(self.nbins(1)))
        index = index % len(self)
        return self.GetBinContent(index + 1)

    def yerravg(self, index=None):

        if index is None:
            return (self.yerravg(i) for i in xrange(self.nbins(1)))
        index = index % len(self)
        return self.GetBinError(index + 1)

    def yerr(self, index=None):

        if index is None:
            return ((self.yerrl(i), self.yerrh(i))
                    for i in xrange(self.nbins(1)))
        index = index % len(self)
        return (self.yerrl(index), self.yerrh(index))

    def GetMaximum(self, **kwargs):

        return self.maximum(**kwargs)

    def maximum(self, include_error=False):

        if not include_error:
            return self.ROOT_base.GetMaximum(self)
        clone = self.Clone()
        for i in xrange(clone.GetNbinsX()):
            clone.SetBinContent(
                i + 1, clone.GetBinContent(i + 1) + clone.GetBinError(i + 1))
        return clone.maximum()

    def GetMinimum(self, **kwargs):

        return self.minimum(**kwargs)

    def minimum(self, include_error=False):

        if not include_error:
            return self.ROOT_base.GetMinimum(self)
        clone = self.Clone()
        for i in xrange(clone.GetNbinsX()):
            clone.SetBinContent(
                i + 1, clone.GetBinContent(i + 1) - clone.GetBinError(i + 1))
        return clone.minimum()

    def expectation(self, startbin=0, endbin=None):

        if endbin is not None and endbin < startbin:
            raise DomainError("endbin should be greated than startbin")
        if endbin is None:
            endbin = len(self) - 1
        expect = 0.
        norm = 0.
        for index in xrange(startbin, endbin + 1):
            val = self[index]
            expect += val * self.x(index)
            norm += val
        if norm > 0:
            return expect / norm
        else:
            return (self.xedges(endbin + 1) + self.xedges(startbin)) / 2

    def _content(self):

        return self.y()

    def _error_content(self):

        return self.yerravg()

    def __getitem__(self, index):

        """
        if type(index) is slice:
            return self._content()[index]
        """
        self._range_check(index)
        return self.y(index)

    def __getslice__(self, i, j):
        # TODO: getslice is deprecated.  getitem should accept slice objects.
        return list(self)[i:j]

    def __setitem__(self, index, value):

        self._range_check(index)
        self.SetBinContent(index + 1, value)


class _Hist2D(_HistBase):

    DIM = 2

    def x(self, index=None):

        return self._centers(1, index)

    def xerravg(self, index=None):

        return self._erravg(1, index)

    def xerrl(self, index=None):

        return self._erravg(1, index)

    def xerrh(self, index=None):

        return self._erravg(1, index)

    def xerr(self, index=None):

        return self._err(1, index)

    def xwidth(self, index=None):

        return self._width(1, index)

    def xedgesl(self, index=None):

        return self._edgesl(1, index)

    def xedgesh(self, index=None):

        return self._edgesh(1, index)

    def xedges(self, index=None):

        return self._edges(1, index)

    def y(self, index=None):

        return self._centers(2, index)

    def yerravg(self, index=None):

        return self._erravg(2, index)

    def yerrl(self, index=None):

        return self._erravg(2, index)

    def yerrh(self, index=None):

        return self._erravg(2, index)

    def yerr(self, index=None):

        return self._err(2, index)

    def ywidth(self, index=None):

        return self._width(2, index)

    def yedgesl(self, index=None):

        return self._edgesl(2, index)

    def yedgesh(self, index=None):

        return self._edgesh(2, index)

    def yedges(self, index=None):

        return self._edges(2, index)

    def zerrh(self, index=None):

        return self.zerravg(index)

    def zerrl(self, index=None):

        return self.zerravg(index)

    def z(self, ix=None, iy=None):

        if ix is None and iy is None:
            return [[self.z(ix, iy)
                    for iy in xrange(self.nbins(2))]
                    for ix in xrange(self.nbins(1))]
        ix = ix % self.nbins(1)
        iy = iy % self.nbins(2)
        return self.GetBinContent(ix + 1, iy + 1)

    def zerravg(self, ix=None, iy=None):

        if ix is None and iy is None:
            return [[self.zerravg(ix, iy)
                    for iy in xrange(self.nbins(2))]
                    for ix in xrange(self.nbins(1))]
        ix = ix % self.nbins(1)
        iy = iy % self.nbins(2)
        return self.GetBinError(ix + 1, iy + 1)

    def zerr(self, ix=None, iy=None):

        if ix is None and iy is None:
            return [[(self.zerravg(ix, iy), self.zerravg(ix, iy))
                    for iy in xrange(self.nbins(2))]
                    for ix in xrange(self.nbins(1))]
        ix = ix % self.nbins(1)
        iy = iy % self.nbins(2)
        return (self.GetBinError(ix + 1, iy + 1),
                self.GetBinError(ix + 1, iy + 1))

    def _content(self):

        return self.z()

    def _error_content(self):

        return self.zerravg()

    def __getitem__(self, index):

        if isinstance(index, tuple):
            # support indexing like h[1, 2]
            return self.z(*index)
        self._range_check(index)
        a = ObjectProxy([
            self.GetBinContent(index + 1, j)
                for j in xrange(1, self.GetNbinsY() + 1)])
        a.__setposthook__('__setitem__', self._setitem(index))
        return a

    def __setitem__(self, index, value):

        ix, iy = index
        self.SetBinContent(ix + 1, iy + 1, value)

    def _setitem(self, i):

        def __setitem(j, value):
            self.SetBinContent(i + 1, j + 1, value)
        return __setitem

    def ravel(self):
        """
        Convert 2D histogram into 1D histogram with the y-axis repeated along
        the x-axis, similar to NumPy's ravel().
        """
        nbinsx = self.nbins(1)
        nbinsy = self.nbins(2)
        left_edge = self.xedgesl(0)
        right_edge = self.xedgesh(-1)
        out = Hist(nbinsx * nbinsy,
                   left_edge, nbinsy * (right_edge - left_edge) + left_edge,
                   type=self.TYPE,
                   title=self.title,
                   **self.decorators)
        for i in xrange(nbinsx):
            for j in xrange(nbinsy):
                out[i * nbinsy + j] = self[i, j]
                out.SetBinError(
                    i * nbinsy + j + 1,
                    self.GetBinError(i + 1, j + 1))
        return out


class _Hist3D(_HistBase):

    DIM = 3

    def x(self, index=None):

        return self._centers(1, index)

    def xerravg(self, index=None):

        return self._erravg(1, index)

    def xerrl(self, index=None):

        return self._erravg(1, index)

    def xerrh(self, index=None):

        return self._erravg(1, index)

    def xerr(self, index=None):

        return self._err(1, index)

    def xwidth(self, index=None):

        return self._width(1, index)

    def xedgesl(self, index=None):

        return self._edgesl(1, index)

    def xedgesh(self, index=None):

        return self._edgesh(1, index)

    def xedges(self, index=None):

        return self._edges(1, index)

    def y(self, index=None):

        return self._centers(2, index)

    def yerravg(self, index=None):

        return self._erravg(2, index)

    def yerrl(self, index=None):

        return self._erravg(2, index)

    def yerrh(self, index=None):

        return self._erravg(2, index)

    def yerr(self, index=None):

        return self._err(2, index)

    def ywidth(self, index=None):

        return self._width(2, index)

    def yedgesl(self, index=None):

        return self._edgesl(2, index)

    def yedgesh(self, index=None):

        return self._edgesh(2, index)

    def yedges(self, index=None):

        return self._edges(2, index)

    def z(self, index=None):

        return self._centers(3, index)

    def zerravg(self, index=None):

        return self._erravg(3, index)

    def zerrl(self, index=None):

        return self._erravg(3, index)

    def zerrh(self, index=None):

        return self._erravg(3, index)

    def zerr(self, index=None):

        return self._err(3, index)

    def zwidth(self, index=None):

        return self._width(3, index)

    def zedgesl(self, index=None):

        return self._edgesl(3, index)

    def zedgesh(self, index=None):

        return self._edgesh(3, index)

    def zedges(self, index=None):

        return self._edges(3, index)

    def werrh(self, index=None):

        return self.werravg(index)

    def werrl(self, index=None):

        return self.werravg(index)

    def w(self, ix=None, iy=None, iz=None):

        if ix is None and iy is None and iz is None:
            return [[[self.w(ix, iy, iz)
                    for iz in xrange(self.nbins(3))]
                    for iy in xrange(self.nbins(2))]
                    for ix in xrange(self.nbins(1))]
        ix = ix % self.nbins(1)
        iy = iy % self.nbins(2)
        iz = iz % self.nbins(3)
        return self.GetBinContent(ix + 1, iy + 1, iz + 1)

    def werravg(self, ix=None, iy=None, iz=None):

        if ix is None and iy is None and iz is None:
            return [[[self.werravg(ix, iy, iz)
                    for iz in xrange(self.nbins(3))]
                    for iy in xrange(self.nbins(2))]
                    for ix in xrange(self.nbins(1))]
        ix = ix % self.nbins(1)
        iy = iy % self.nbins(2)
        iz = iz % self.nbins(3)
        return self.GetBinError(ix + 1, iy + 1, iz + 1)

    def werr(self, ix=None, iy=None, iz=None):

        if ix is None and iy is None and iz is None:
            return [[[(self.werravg(ix, iy, iz), self.werravg(ix, iy, iz))
                    for iz in xrange(self.nbins(3))]
                    for iy in xrange(self.nbins(2))]
                    for ix in xrange(self.nbins(1))]
        ix = ix % self.nbins(1)
        iy = iy % self.nbins(2)
        iz = iz % self.nbins(3)
        return (self.GetBinError(ix + 1, iy + 1, iz + 1),
                self.GetBinError(ix + 1, iy + 1, iz + 1))

    def _content(self):

        return [[[
            self.GetBinContent(i, j, k)
                for i in xrange(1, self.GetNbinsX() + 1)]
                    for j in xrange(1, self.GetNbinsY() + 1)]
                        for k in xrange(1, self.GetNbinsZ() + 1)]

    def _error_content(self):

        return [[[
            self.GetBinError(i, j, k)
                for i in xrange(1, self.GetNbinsX() + 1)]
                    for j in xrange(1, self.GetNbinsY() + 1)]
                        for k in xrange(1, self.GetNbinsZ() + 1)]

    def __getitem__(self, index):

        if isinstance(index, tuple):
            # support indexing like h[1,2,1]
            return self.w(*index)
        self._range_check(index)
        out = []
        for j in xrange(1, self.GetNbinsY() + 1):
            a = ObjectProxy([
                self.GetBinContent(index + 1, j, k)
                    for k in xrange(1, self.GetNbinsZ() + 1)])
            a.__setposthook__('__setitem__', self._setitem(index, j - 1))
            out.append(a)
        return out

    def __setitem__(self, index, value):

        ix, iy, iz = index
        self.SetBinContent(ix + 1, iy + 1, iz + 1, value)

    def _setitem(self, i, j):

        def __setitem(k, value):
            self.SetBinContent(i + 1, j + 1, k + 1, value)
        return __setitem


def _Hist_class(type='F'):

    type = type.upper()
    if type not in _HistBase.TYPES:
        raise TypeError("No histogram available with bin type %s" % type)
    rootclass = _HistBase.TYPES[type][0]

    class Hist(_Hist, rootclass):
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
        raise TypeError("No histogram available with bin type %s" % type)
    rootclass = _HistBase.TYPES[type][1]

    class Hist2D(_Hist2D, rootclass):
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
        raise TypeError("No histogram available with bin type %s" % type)
    rootclass = _HistBase.TYPES[type][2]

    class Hist3D(_Hist3D, rootclass):
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
                            for n in xrange(params[0]['nbins'] + 1)]
                if params[1]['bins'] is None:
                    step = ((params[1]['high'] - params[1]['low'])
                        / float(params[1]['nbins']))
                    params[1]['bins'] = [
                        params[1]['low'] + n * step
                            for n in xrange(params[1]['nbins'] + 1)]
                if params[2]['bins'] is None:
                    step = ((params[2]['high'] - params[2]['low'])
                        / float(params[2]['nbins']))
                    params[2]['bins'] = [
                        params[2]['low'] + n * step
                            for n in xrange(params[2]['nbins'] + 1)]
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


class Hist(_Hist):
    """
    Returns a 1-dimensional Hist object which inherits from the associated
    ROOT.TH1* class (where * is C, S, I, F, or D depending on the type
    keyword argument)
    """
    @classmethod
    def dynamic_cls(cls, type):

        return _HIST_CLASSES_1D[type]

    def __new__(cls, *args, **kwargs):

        type = kwargs.pop('type', 'F').upper()
        return cls.dynamic_cls(type)(
            *args, **kwargs)


class Hist2D(_Hist2D):
    """
    Returns a 2-dimensional Hist object which inherits from the associated
    ROOT.TH1* class (where * is C, S, I, F, or D depending on the type
    keyword argument)
    """
    @classmethod
    def dynamic_cls(cls, type):

        return _HIST_CLASSES_2D[type]

    def __new__(cls, *args, **kwargs):

        type = kwargs.pop('type', 'F').upper()
        return cls.dynamic_cls(type)(
            *args, **kwargs)


class Hist3D(_Hist3D):
    """
    Returns a 3-dimensional Hist object which inherits from the associated
    ROOT.TH1* class (where * is C, S, I, F, or D depending on the type
    keyword argument)
    """
    @classmethod
    def dynamic_cls(cls, type):

        return _HIST_CLASSES_3D[type]

    def __new__(cls, *args, **kwargs):

        type = kwargs.pop('type', 'F').upper()
        return cls.dynamic_cls(type)(
            *args, **kwargs)


from rootpy import ROOT_VERSION
if ROOT_VERSION >= 52800:

    @snake_case_methods
    class Efficiency(Plottable, NamedObject, QROOT.TEfficiency):

        def __init__(self, passed, total, name=None, title=None, **kwargs):

            if dim(passed) != 1 or dim(total) != 1:
                raise TypeError(
                        "histograms must be 1 dimensional")
            if len(passed) != len(total):
                raise ValueError(
                        "histograms must have the same number of bins")
            if list(passed.xedges()) != list(total.xedges()):
                raise ValueError(
                        "histograms do not have the same bin boundaries")

            super(Efficiency, self).__init__(
                len(total), total.xedgesl(0), total.xedgesh(-1),
                name=name, title=title)

            self.passed = passed.Clone()
            self.total = total.Clone()
            self.SetPassedHistogram(self.passed, 'f')
            self.SetTotalHistogram(self.total, 'f')
            self._post_init(**kwargs)

        def __len__(self):

            return len(self.total)

        def __getitem__(self, bin):

            return self.GetEfficiency(bin + 1)

        def __add__(self, other):

            copy = self.Clone()
            copy.Add(other)
            return copy

        def __iadd__(self, other):

            super(Efficiency, self).Add(self, other)
            return self

        def __iter__(self):

            for bin in xrange(len(self)):
                yield self[bin]

        def errors(self):

            for bin in xrange(len(self)):
                yield (self.GetEfficiencyErrorLow(bin + 1),
                       self.GetEfficiencyErrorUp(bin + 1))

        def GetGraph(self):

            graph = Graph(len(self))
            for index, (bin, effic, (low, up)) in enumerate(
                    zip(xrange(len(self)), iter(self), self.errors())):
                graph.SetPoint(index, self.total.x(bin), effic)
                xerror = self.total.xwidth(bin) / 2.
                graph.SetPointError(index, xerror, xerror, low, up)
            return graph

        @property
        def painted_graph(self):
            """
            Returns the painted graph for a TEfficiency, or if it isn't
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


class HistStack(Plottable, NamedObject, QROOT.THStack):

    def __init__(self, name=None, title=None, hists=None, **kwargs):

        super(HistStack, self).__init__(name=name, title=title)
        self._post_init(hists=hists, **kwargs)

    def _post_init(self, hists=None, **kwargs):

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
                self.sum = hist.Clone()
            elif dim(self) != dim(hist):
                raise TypeError(
                    "Dimension of histogram does not match dimension "
                    "of already contained histograms")
            else:
                self.sum += hist
            self.hists.append(hist)
            super(HistStack, self).Add(hist, hist.drawstyle)
        else:
            raise TypeError(
                "Only 1D and 2D histograms are supported")

    def __add__(self, other):

        if not isinstance(other, HistStack):
            raise TypeError(
                "Addition not supported for HistStack and %s" %
                other.__class__.__name__)
        clone = HistStack()
        for hist in self:
            clone.Add(hist)
        for hist in other:
            clone.Add(hist)
        return clone

    def __iadd__(self, other):

        if not isinstance(other, HistStack):
            raise TypeError(
                "Addition not supported for HistStack and %s" %
                other.__class__.__name__)
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

    def __cmp__(self, other):

        diff = self.maximum() - other.maximum()
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
        if start != None and end != None:
            for hist in self:
                integral += hist.Integral(start, end)
        else:
            for hist in self:
                integral += hist.Integral()
        return integral

    def lowerbound(self, axis=1):

        if not self:
            return None  # negative infinity
        return min(hist.lowerbound(axis=axis) for hist in self)

    def upperbound(self, axis=1):

        if not self:
            return ()  # positive infinity
        return max(hist.upperbound(axis=axis) for hist in self)

    def GetMaximum(self, **kwargs):

        return self.maximum(**kwargs)

    def maximum(self, **kwargs):

        if not self:
            return 0
        return self.sum.GetMaximum(**kwargs)

    def GetMinimum(self, **kwargs):

        return self.minimum(**kwargs)

    def minimum(self, **kwargs):

        if not self:
            return 0
        return self.sum.GetMinimum(**kwargs)

    def Clone(self, newName=None):

        clone = HistStack(name=newName,
                          title=self.GetTitle(),
                          **self.decorators)
        for hist in self:
            clone.Add(hist.Clone())
        return clone


def FillHistogram(data, *args, **kwargs):
    """
    Create an histogram filling it with data. It takes as input the same inputs
    as the Hist class. If the number of bins and the ranges are not specified
    they are automatically deduced using the autobinning function using the
    method specified by the binning argument.
    It works only for 1D histograms.
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

