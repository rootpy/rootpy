from .. import QROOT, log; log = log[__name__]
from .hist import _Hist, _Hist2D, _Hist3D
from ..core import Object


class _ProfileBase(object):
    pass


class Profile(_ProfileBase, _Hist, QROOT.TProfile):

    def __init__(self, *args, **kwargs):

        if 'name' in kwargs:
            name = kwargs['name']
            del kwargs['name']
        else:
            name = None
        if 'title' in kwargs:
            title = kwargs['title']
            del kwargs['title']
        else:
            title = None
        if 'option' in kwargs:
            option = kwargs['option']
            del kwargs['option']
        else:
            option = ""

        params, args = self._parse_args(args, ignore_extras=True)
        if args and len(args) != 2:
            raise TypeError("Did not receive expected number of arguments")
        args = list(args)
        args.append(option)

        if params[0]['bins'] is None:
            Object.__init__(self, name, title,
                params[0]['nbins'], params[0]['low'], params[0]['high'],
                *args)
        else:
            Object.__init__(self, name, title,
                params[0]['nbins'], array('d', params[0]['bins']),
                *args)

        self._post_init(**kwargs)


class Profile2D(_ProfileBase, _Hist2D, QROOT.TProfile2D):

    def __init__(self, *args, **kwargs):

        if 'name' in kwargs:
            name = kwargs['name']
            del kwargs['name']
        else:
            name = None
        if 'title' in kwargs:
            title = kwargs['title']
            del kwargs['title']
        else:
            title = None
        if 'option' in kwargs:
            option = kwargs['option']
            del kwargs['option']
        else:
            option = ""

        params, args = self._parse_args(args, ignore_extras=True)
        if args and len(args) != 2:
            raise TypeError("Did not receive expected number of arguments")
        args = list(args)
        args.append(option)

        if params[0]['bins'] is None and params[1]['bins'] is None:
            Object.__init__(self, name, title,
                params[0]['nbins'], params[0]['low'], params[0]['high'],
                params[1]['nbins'], params[1]['low'], params[1]['high'],
                *args)
        elif params[0]['bins'] is None and params[1]['bins'] is not None:
            Object.__init__(self, name, title,
                params[0]['nbins'], params[0]['low'], params[0]['high'],
                params[1]['nbins'], array('d', params[1]['bins']),
                *args)
        elif params[0]['bins'] is not None and params[1]['bins'] is None:
            Object.__init__(self, name, title,
                params[0]['nbins'], array('d', params[0]['bins']),
                params[1]['nbins'], params[1]['low'], params[1]['high'],
                *args)
        else:
            Object.__init__(self, name, title,
                params[0]['nbins'], array('d', params[0]['bins']),
                params[1]['nbins'], array('d', params[1]['bins']),
                *args)

        self._post_init(**kwargs)


class Profile3D(_ProfileBase, _Hist3D, QROOT.TProfile3D):

    def __init__(self, *args, **kwargs):

        if 'name' in kwargs:
            name = kwargs['name']
            del kwargs['name']
        else:
            name = None
        if 'title' in kwargs:
            title = kwargs['title']
            del kwargs['title']
        else:
            title = None
        if 'option' in kwargs:
            option = kwargs['option']
            del kwargs['option']
        else:
            option = ""

        # Profile3D does not support t_low, t_up
        params = self._parse_args(args)

        # ROOT is missing constructors for TH3...
        if params[0]['bins'] is None and \
           params[1]['bins'] is None and \
           params[2]['bins'] is None:
            Object.__init__(self, name, title,
                params[0]['nbins'], params[0]['low'], params[0]['high'],
                params[1]['nbins'], params[1]['low'], params[1]['high'],
                params[2]['nbins'], params[2]['low'], params[2]['high'],
                option)
        else:
            if params[0]['bins'] is None:
                step = (params[0]['high'] - params[0]['low'])\
                    / float(params[0]['nbins'])
                params[0]['bins'] = [
                    params[0]['low'] + n * step
                        for n in xrange(params[0]['nbins'] + 1)]
            if params[1]['bins'] is None:
                step = (params[1]['high'] - params[1]['low'])\
                    / float(params[1]['nbins'])
                params[1]['bins'] = [
                    params[1]['low'] + n * step
                        for n in xrange(params[1]['nbins'] + 1)]
            if params[2]['bins'] is None:
                step = (params[2]['high'] - params[2]['low'])\
                    / float(params[2]['nbins'])
                params[2]['bins'] = [
                    params[2]['low'] + n * step
                        for n in xrange(params[2]['nbins'] + 1)]
            Object.__init__(self, name, title,
                params[0]['nbins'], array('d', params[0]['bins']),
                params[1]['nbins'], array('d', params[1]['bins']),
                params[2]['nbins'], array('d', params[2]['bins']),
                option)

        self._post_init(**kwargs)

