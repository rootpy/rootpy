# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
from __future__ import absolute_import

import os
import re
import shutil
from glob import glob

import ROOT

from . import log; log = log[__name__]
from ...extern.six import string_types
from ...memory.keepalive import keepalive
from ...utils.silence import silence_sout_serr
from ...utils.path import mkdir_p
from ...context import (
    do_nothing, working_directory, preserve_current_directory)
from ...io import root_open
from ... import asrootpy
from . import Channel, Measurement, HistoSys, OverallSys

__all__ = [
    'make_channel',
    'make_measurement',
    'make_workspace',
    'measurements_from_xml',
    'write_measurement',
    'patch_xml',
    'split_norm_shape',
]


def make_channel(name, samples, data=None, verbose=False):
    """
    Create a Channel from a list of Samples
    """
    if verbose:
        llog = log['make_channel']
        llog.info("creating channel {0}".format(name))
    # avoid segfault if name begins with a digit by using "channel_" prefix
    chan = Channel('channel_{0}'.format(name))
    chan.SetStatErrorConfig(0.05, "Poisson")

    if data is not None:
        if verbose:
            llog.info("setting data")
        chan.SetData(data)

    for sample in samples:
        if verbose:
            llog.info("adding sample {0}".format(sample.GetName()))
        chan.AddSample(sample)

    return chan


def make_measurement(name,
                     channels,
                     lumi=1.0, lumi_rel_error=0.1,
                     output_prefix='./histfactory',
                     POI=None,
                     const_params=None,
                     verbose=False):
    """
    Create a Measurement from a list of Channels
    """
    if verbose:
        llog = log['make_measurement']
        llog.info("creating measurement {0}".format(name))

    if not isinstance(channels, (list, tuple)):
        channels = [channels]

    # Create the measurement
    meas = Measurement('measurement_{0}'.format(name), '')
    meas.SetOutputFilePrefix(output_prefix)
    if POI is not None:
        if isinstance(POI, string_types):
            if verbose:
                llog.info("setting POI {0}".format(POI))
            meas.SetPOI(POI)
        else:
            if verbose:
                llog.info("adding POIs {0}".format(', '.join(POI)))
            for p in POI:
                meas.AddPOI(p)

    if verbose:
        llog.info("setting lumi={0:f} +/- {1:f}".format(lumi, lumi_rel_error))
    meas.lumi = lumi
    meas.lumi_rel_error = lumi_rel_error

    for channel in channels:
        if verbose:
            llog.info("adding channel {0}".format(channel.GetName()))
        meas.AddChannel(channel)

    if const_params is not None:
        if verbose:
            llog.info("adding constant parameters {0}".format(
                ', '.join(const_params)))
        for param in const_params:
            meas.AddConstantParam(param)

    return meas


def make_workspace(measurement, channel=None, name=None, silence=False):
    """
    Create a workspace containing the model for a measurement

    If `channel` is None then include all channels in the model

    If `silence` is True, then silence HistFactory's output on
    stdout and stderr.
    """
    context = silence_sout_serr if silence else do_nothing
    with context():
        hist2workspace = ROOT.RooStats.HistFactory.HistoToWorkspaceFactoryFast(
            measurement)
        if channel is not None:
            workspace = hist2workspace.MakeSingleChannelModel(
                measurement, channel)
        else:
            workspace = hist2workspace.MakeCombinedModel(measurement)
    workspace = asrootpy(workspace)
    keepalive(workspace, measurement)
    if name is not None:
        workspace.SetName('workspace_{0}'.format(name))
    return workspace


def measurements_from_xml(filename,
                          collect_histograms=True,
                          cd_parent=False,
                          silence=False):
    """
    Read in a list of Measurements from XML
    """
    if not os.path.isfile(filename):
        raise OSError("the file {0} does not exist".format(filename))
    silence_context = silence_sout_serr if silence else do_nothing

    filename = os.path.abspath(os.path.normpath(filename))

    if cd_parent:
        xml_directory = os.path.dirname(filename)
        parent = os.path.abspath(os.path.join(xml_directory, os.pardir))
        cd_context = working_directory
    else:
        parent = None
        cd_context = do_nothing

    log.info("parsing XML in {0} ...".format(filename))
    with cd_context(parent):
        parser = ROOT.RooStats.HistFactory.ConfigParser()
        with silence_context():
            measurements_vect = parser.GetMeasurementsFromXML(filename)
        # prevent measurements_vect from being garbage collected
        ROOT.SetOwnership(measurements_vect, False)
        measurements = []
        for m in measurements_vect:
            if collect_histograms:
                with silence_context():
                    m.CollectHistograms()
            measurements.append(asrootpy(m))
    return measurements


def write_measurement(measurement,
                      root_file=None,
                      xml_path=None,
                      output_path=None,
                      output_suffix=None,
                      write_workspaces=False,
                      apply_xml_patches=True,
                      silence=False):
    """
    Write a measurement and RooWorkspaces for all contained channels
    into a ROOT file and write the XML files into a directory.

    Parameters
    ----------

    measurement : HistFactory::Measurement
        An asrootpy'd ``HistFactory::Measurement`` object

    root_file : ROOT TFile or string, optional (default=None)
        A ROOT file or string file name. The measurement and workspaces
        will be written to this file. If ``root_file is None`` then a
        new file will be created with the same name as the measurement and
        with the prefix ``ws_``.

    xml_path : string, optional (default=None)
        A directory path to write the XML into. If None, a new directory with
        the same name as the measurement and with the prefix ``xml_`` will be
        created.

    output_path : string, optional (default=None)
        If ``root_file is None``, create the ROOT file under this path.
        If ``xml_path is None``, create the XML directory under this path.

    output_suffix : string, optional (default=None)
        If ``root_file is None`` then a new file is created with the same name
        as the measurement and with the prefix ``ws_``. ``output_suffix`` will
        append a suffix to this file name (before the .root extension).
        If ``xml_path is None``, then a new directory is created with the
        same name as the measurement and with the prefix ``xml_``.
        ``output_suffix`` will append a suffix to this directory name.

    write_workspaces : bool, optional (default=False)
        If True then also write a RooWorkspace for each channel and for all
        channels combined.

    apply_xml_patches : bool, optional (default=True)
        Apply fixes on the output of ``Measurement::PrintXML()`` to avoid known
        HistFactory bugs. Some of the patches assume that the ROOT file
        containing the histograms will exist one directory level up from the
        XML and that hist2workspace, or any tool that later reads the XML will
        run from that same directory containing the ROOT file.

    silence : bool, optional (default=False)
        If True then capture and silence all stdout/stderr output from
        HistFactory.

    """
    context = silence_sout_serr if silence else do_nothing

    output_name = measurement.name
    if output_suffix is not None:
        output_name += '_{0}'.format(output_suffix)
    output_name = output_name.replace(' ', '_')

    if xml_path is None:
        xml_path = 'xml_{0}'.format(output_name)
        if output_path is not None:
            xml_path = os.path.join(output_path, xml_path)

    if not os.path.exists(xml_path):
        mkdir_p(xml_path)

    if root_file is None:
        root_file = 'ws_{0}.root'.format(output_name)
        if output_path is not None:
            root_file = os.path.join(output_path, root_file)

    own_file = False
    if isinstance(root_file, string_types):
        root_file = root_open(root_file, 'recreate')
        own_file = True

    with preserve_current_directory():
        root_file.cd()

        log.info("writing histograms and measurement in {0} ...".format(
            root_file.GetName()))
        with context():
            measurement.writeToFile(root_file)
        # get modified measurement
        out_m = root_file.Get(measurement.name)
        log.info("writing XML in {0} ...".format(xml_path))
        with context():
            out_m.PrintXML(xml_path)

        if write_workspaces:
            log.info("writing combined model in {0} ...".format(
                root_file.GetName()))
            workspace = make_workspace(measurement, silence=silence)
            workspace.Write()
            for channel in measurement.channels:
                log.info("writing model for channel `{0}` in {1} ...".format(
                    channel.name, root_file.GetName()))
                workspace = make_workspace(
                    measurement, channel=channel, silence=silence)
                workspace.Write()

    if apply_xml_patches:
        # patch the output XML to avoid HistFactory bugs
        patch_xml(glob(os.path.join(xml_path, '*.xml')),
                  root_file=os.path.basename(root_file.GetName()))

    if own_file:
        root_file.Close()


def patch_xml(files, root_file=None, float_precision=3):
    """
    Apply patches to HistFactory XML output from PrintXML
    """
    if float_precision < 0:
        raise ValueError("precision must be greater than 0")

    def fix_path(match):
        path = match.group(1)
        if path:
            head, tail = os.path.split(path)
            new_path = os.path.join(os.path.basename(head), tail)
        else:
            new_path = ''
        return '<Input>{0}</Input>'.format(new_path)

    for xmlfilename in files:
        xmlfilename = os.path.abspath(os.path.normpath(xmlfilename))
        patched_xmlfilename = '{0}.tmp'.format(xmlfilename)
        log.info("patching {0} ...".format(xmlfilename))
        fin = open(xmlfilename, 'r')
        fout = open(patched_xmlfilename, 'w')
        for line in fin:
            if root_file is not None:
                line = re.sub(
                    'InputFile="[^"]*"',
                    'InputFile="{0}"'.format(root_file), line)
            line = line.replace(
                '<StatError Activate="True"  InputFile=""  '
                'HistoName=""  HistoPath=""  />',
                '<StatError Activate="True" />')
            line = re.sub(
                '<Combination OutputFilePrefix="(\S*)" >',
                '<Combination OutputFilePrefix="hist2workspace" >', line)
            line = re.sub('\w+=""', '', line)
            line = re.sub('\s+/>', ' />', line)
            line = re.sub('(\S)\s+</', r'\1</', line)
            # HistFactory bug:
            line = re.sub('InputFileHigh="\S+"', '', line)
            line = re.sub('InputFileLow="\S+"', '', line)
            # HistFactory bug:
            line = line.replace(
                '<ParamSetting Const="True"></ParamSetting>', '')
            # chop off floats to desired precision
            line = re.sub(
                r'"(\d*\.\d{{{0:d},}})"'.format(float_precision + 1),
                lambda x: '"{0}"'.format(
                    str(round(float(x.group(1)), float_precision))),
                line)
            line = re.sub('"\s\s+(\S)', r'" \1', line)
            line = re.sub('<Input>(.*)</Input>', fix_path, line)
            fout.write(line)
        fin.close()
        fout.close()
        shutil.move(patched_xmlfilename, xmlfilename)
        if not os.path.isfile(os.path.join(
                              os.path.dirname(xmlfilename),
                              'HistFactorySchema.dtd')):
            rootsys = os.getenv('ROOTSYS', None)
            if rootsys is not None:
                dtdfile = os.path.join(rootsys, 'etc/HistFactorySchema.dtd')
                target = os.path.dirname(xmlfilename)
                if os.path.isfile(dtdfile):
                    log.info("copying {0} to {1} ...".format(dtdfile, target))
                    shutil.copy(dtdfile, target)
                else:
                    log.warning("{0} does not exist".format(dtdfile))
            else:
                log.warning(
                    "$ROOTSYS is not set so cannot find HistFactorySchema.dtd")


def split_norm_shape(histosys, nominal_hist):
    """
    Split a HistoSys into normalization (OverallSys) and shape (HistoSys)
    components.

    It is recommended to use OverallSys as much as possible, which tries to
    enforce continuity up to the second derivative during
    interpolation/extrapolation. So, if there is indeed a shape variation, then
    factorize it into shape and normalization components.
    """
    up = histosys.GetHistoHigh()
    dn = histosys.GetHistoLow()
    up = up.Clone(name=up.name + '_shape')
    dn = dn.Clone(name=dn.name + '_shape')
    n_nominal = nominal_hist.integral(overflow=True)
    n_up = up.integral(overflow=True)
    n_dn = dn.integral(overflow=True)
    if n_up != 0:
        up.Scale(n_nominal / n_up)
    if n_dn != 0:
        dn.Scale(n_nominal / n_dn)
    shape = HistoSys(histosys.GetName(), low=dn, high=up)
    norm = OverallSys(histosys.GetName(),
                      low=n_dn / n_nominal if n_nominal != 0 else 1.,
                      high=n_up / n_nominal if n_nominal != 0 else 1.)
    return norm, shape
