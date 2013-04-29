# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
from array import array
from .. import QROOT, log; log = log[__name__]
from .hist import _Hist, _Hist2D, _Hist3D


class _ProfileBase(object):
    pass


class Profile(_ProfileBase, _Hist, QROOT.TProfile):

    def __init__(self, *args, **kwargs):

        if 'option' in kwargs:
            option = kwargs['option']
            del kwargs['option']
        else:
            option = ""

        params, args = self._parse_args(args, ignore_extras=True)
        if args:
            if len(args) != 2:
                raise TypeError("Did not receive expected number of arguments")
            low, high = args
            if low >= high:
                raise ValueError(
                    "Upper bound (you gave %f) must be greater than lower "
                    "bound (you gave %f)" % (float(low), float(high)))
        args = list(args)
        args.append(option)

        if params[0]['bins'] is None:
            super(Profile, self).__init__(
                params[0]['nbins'], params[0]['low'], params[0]['high'],
                *args, **kwargs)
        else:
            super(Profile, self).__init__(
                params[0]['nbins'], array('d', params[0]['bins']),
                *args, **kwargs)


class Profile2D(_ProfileBase, _Hist2D, QROOT.TProfile2D):

    def __init__(self, *args, **kwargs):

        if 'option' in kwargs:
            option = kwargs['option']
            del kwargs['option']
        else:
            option = ""

        params, args = self._parse_args(args, ignore_extras=True)
        if args:
            if len(args) != 2:
                raise TypeError("Did not receive expected number of arguments")
            low, high = args
            if low >= high:
                raise ValueError(
                    "Upper bound (you gave %f) must be greater than lower "
                    "bound (you gave %f)" % (float(low), float(high)))
        args = list(args)
        args.append(option)

        if params[0]['bins'] is None and params[1]['bins'] is None:
            super(Profile2D, self).__init__(
                params[0]['nbins'], params[0]['low'], params[0]['high'],
                params[1]['nbins'], params[1]['low'], params[1]['high'],
                *args, **kwargs)
        elif params[0]['bins'] is None and params[1]['bins'] is not None:
            super(Profile2D, self).__init__(
                params[0]['nbins'], params[0]['low'], params[0]['high'],
                params[1]['nbins'], array('d', params[1]['bins']),
                *args, **kwargs)
        elif params[0]['bins'] is not None and params[1]['bins'] is None:
            super(Profile2D, self).__init__(
                params[0]['nbins'], array('d', params[0]['bins']),
                params[1]['nbins'], params[1]['low'], params[1]['high'],
                *args, **kwargs)
        else:
            super(Profile2D, self).__init__(
                params[0]['nbins'], array('d', params[0]['bins']),
                params[1]['nbins'], array('d', params[1]['bins']),
                *args, **kwargs)


class Profile3D(_ProfileBase, _Hist3D, QROOT.TProfile3D):

    def __init__(self, *args, **kwargs):

        if 'option' in kwargs:
            option = kwargs['option']
            del kwargs['option']
        else:
            option = ""

        # Profile3D does not support t_low, t_up
        params = self._parse_args(args)

        # ROOT is missing constructors for TH3...
        if (params[0]['bins'] is None and
            params[1]['bins'] is None and
            params[2]['bins'] is None):
            super(Profile3D, self).__init__(
                params[0]['nbins'], params[0]['low'], params[0]['high'],
                params[1]['nbins'], params[1]['low'], params[1]['high'],
                params[2]['nbins'], params[2]['low'], params[2]['high'],
                option, **kwargs)
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
            super(Profile3D, self).__init__(
                params[0]['nbins'], array('d', params[0]['bins']),
                params[1]['nbins'], array('d', params[1]['bins']),
                params[2]['nbins'], array('d', params[2]['bins']),
                option, **kwargs)
