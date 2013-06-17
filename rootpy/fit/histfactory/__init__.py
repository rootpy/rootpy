# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
from .. import log; log = log[__name__]
from ... import ROOT_VERSION, ROOTVersion

MIN_ROOT_VERSION = ROOTVersion(53404)

if ROOT_VERSION >= MIN_ROOT_VERSION:
    from .histfactory import (Data,
                              Sample,
                              HistoSys,
                              HistoFactor,
                              NormFactor,
                              OverallSys,
                              ShapeFactor,
                              ShapeSys,
                              Channel,
                              Measurement)

    from .utils import (make_channel,
                        make_measurement,
                        make_models,
                        make_model,
                        make_workspace)

    __all__ = [
        'Data',
        'Sample',
        'HistoSys',
        'HistoFactor',
        'NormFactor',
        'OverallSys',
        'ShapeFactor',
        'ShapeSys',
        'Channel',
        'Measurement',
        'make_channel',
        'make_measurement',
        'make_models',
        'make_model',
        'make_workspace',
    ]

else:
    import warnings
    warnings.warn(
        "histfactory requires ROOT {0} but you are using {1}".format(
            MIN_ROOT_VERSION, ROOT_VERSION))
    __all__ = []
