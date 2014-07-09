# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
from __future__ import absolute_import

from .. import log; log = log[__name__]
from ... import ROOT_VERSION, ROOTVersion

MIN_ROOT_VERSION = ROOTVersion(53404)

if ROOT_VERSION >= MIN_ROOT_VERSION:
    from .histfactory import (Constraint,
                              Data,
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
                        make_workspace,
                        measurements_from_xml,
                        write_measurement,
                        patch_xml,
                        split_norm_shape)

    __all__ = [
        'Constraint',
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
        'make_workspace',
        'measurements_from_xml',
        'write_measurement',
        'patch_xml',
        'split_norm_shape',
    ]

    from ... import stl

    # generate required dictionaries
    stl.vector('RooStats::HistFactory::HistoSys',
               headers='<vector>;<RooStats/HistFactory/Systematics.h>')
    stl.vector('RooStats::HistFactory::HistoFactor',
               headers='<vector>;<RooStats/HistFactory/Systematics.h>')
    stl.vector('RooStats::HistFactory::NormFactor',
               headers='<vector>;<RooStats/HistFactory/Systematics.h>')
    stl.vector('RooStats::HistFactory::OverallSys',
               headers='<vector>;<RooStats/HistFactory/Systematics.h>')
    stl.vector('RooStats::HistFactory::ShapeFactor',
               headers='<vector>;<RooStats/HistFactory/Systematics.h>')
    stl.vector('RooStats::HistFactory::ShapeSys',
               headers='<vector>;<RooStats/HistFactory/Systematics.h>')
    stl.vector('RooStats::HistFactory::Sample')
    stl.vector('RooStats::HistFactory::Data')
    stl.vector('RooStats::HistFactory::Channel')
    stl.vector('RooStats::HistFactory::Measurement')

else:
    import warnings
    warnings.warn(
        "histfactory requires ROOT {0} but you are using {1}".format(
            MIN_ROOT_VERSION, ROOT_VERSION))
    __all__ = []
