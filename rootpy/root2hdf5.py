# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
"""
This module handles conversion of ROOT's TFile and
contained TTrees into HDF5 format with PyTables
"""
from __future__ import absolute_import

import os
import sys
import tables
import warnings

from .io import root_open, utils, TemporaryFile
from . import log; log = log[__name__]
from .extern.progressbar import ProgressBar, Bar, ETA, Percentage
from .logger.util import check_tty

from root_numpy import tree2rec, RootNumpyUnconvertibleWarning


def _drop_object_col(rec, warn=True):
    # ignore columns of type `object` since PyTables does not support these
    if rec.dtype.hasobject:
        names = []
        fields = rec.dtype.fields
        for name in rec.dtype.names:
            if fields[name][0].kind != 'O':
                names.append(name)
            elif warn:
                log.warning("ignoring unsupported object branch '%s'" % name)
        return rec[names]
    return rec


def convert(rfile, hfile, rpath='', entries=-1, userfunc=None, selection=None):

    isatty = check_tty(sys.stdout)
    if isatty:
        widgets = [Percentage(), ' ', Bar(), ' ', ETA()]

    own_h5file = False
    if isinstance(hfile, basestring):
        hfile = tables.openFile(filename=hfile, mode="w", title="Data")
        own_h5file = True
    own_rootfile = False
    if isinstance(rfile, basestring):
        rfile = root_open(rfile)
        own_rootfile = True

    for dirpath, dirnames, treenames in utils.walk(
            rfile, rpath, class_pattern='TTree'):

        # skip root
        if not dirpath and not treenames:
            continue

        # skip directories w/o trees or subdirs
        if not dirnames and not treenames:
            continue

        where_group = '/' + os.path.dirname(dirpath)
        current_dir = os.path.basename(dirpath)

        if not current_dir:
            group = hfile.root
        else:
            group = hfile.createGroup(where_group, current_dir, "")

        ntrees = len(treenames)
        if ntrees > 1:
            log.info("Will convert %i trees in this directory" % ntrees)
        else:
            log.info("Will convert 1 tree in this directory")

        for treename in treenames:

            input_tree = rfile.Get(os.path.join(dirpath, treename))

            if userfunc is not None:
                tmp_file = TemporaryFile()
                # call user-defined function on tree and get output trees
                log.info("Calling user function on tree '%s'" %
                    input_tree.GetName())
                trees = userfunc(input_tree)

                if not isinstance(trees, list):
                    trees = [trees]

            else:
                trees = [input_tree]
                tmp_file = None

            for tree in trees:

                log.info("Converting tree '%s' with %i entries ..." % (
                    tree.GetName(),
                    tree.GetEntries()))

                total_entries = tree.GetEntries()
                pbar = None
                if isatty and total_entries > 0:
                    pbar = ProgressBar(widgets=widgets, maxval=total_entries)

                if entries <= 0:
                    # read the entire tree
                    if pbar is not None:
                        pbar.start()
                    recarray = tree2rec(tree, selection=selection)
                    recarray = _drop_object_col(recarray)
                    table = hfile.createTable(
                        group, tree.GetName(),
                        recarray, tree.GetTitle())
                    # flush data in the table
                    table.flush()
                    # flush all pending data
                    hfile.flush()
                else:
                    # read the tree in chunks
                    offset = 0
                    while offset < total_entries or offset == 0:
                        if offset > 0:
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore",
                                        RootNumpyUnconvertibleWarning)
                                recarray = tree2rec(tree,
                                        entries=entries,
                                        offset=offset,
                                        selection=selection)
                            recarray = _drop_object_col(recarray, warn=False)
                            table.append(recarray)
                        else:
                            recarray = tree2rec(tree,
                                    entries=entries,
                                    offset=offset,
                                    selection=selection)
                            recarray = _drop_object_col(recarray)
                            if pbar is not None:
                                # start after any output from root_numpy
                                pbar.start()
                            table = hfile.createTable(
                                group, tree.GetName(),
                                recarray, tree.GetTitle())
                        offset += entries
                        if offset <= total_entries and pbar is not None:
                            pbar.update(offset)
                        # flush data in the table
                        table.flush()
                        # flush all pending data
                        hfile.flush()

                if pbar is not None:
                    pbar.finish()

            input_tree.Delete()

            if userfunc is not None:
                for tree in trees:
                    tree.Delete()
                tmp_file.Close()

    if own_h5file:
        hfile.close()
    if own_rootfile:
        rfile.Close()


def main():

    from rootpy.extern.argparse import (ArgumentParser,
            ArgumentDefaultsHelpFormatter, RawTextHelpFormatter)

    class formatter_class(ArgumentDefaultsHelpFormatter,
                          RawTextHelpFormatter):
        pass

    parser = ArgumentParser(formatter_class=formatter_class)
    parser.add_argument('-n', '--entries', type=int, default=1E5,
            help="number of entries to read at once")
    parser.add_argument('-f', '--force', action='store_true', default=False,
            help="overwrite existing output files")
    parser.add_argument('--ext', default='h5',
            help="output file extension")
    parser.add_argument('-c', '--complevel', type=int, default=5,
            choices=range(0, 10),
            help="compression level")
    parser.add_argument('-l', '--complib', default='zlib',
            choices=('zlib', 'lzo', 'bzip2', 'blosc'),
            help="compression algorithm")
    parser.add_argument('-s', '--selection', default=None,
            help="apply a selection on each tree with a cut expression")
    parser.add_argument('--script', default=None,
            help="Python script containing a function with the same name \n"
                 "that will be called on each tree and must return a tree or \n"
                 "list of trees that will be converted instead of the \n"
                 "original tree")
    parser.add_argument('-q', '--quiet', action='store_true', default=False,
            help="suppress all warnings")
    parser.add_argument('files', nargs='+')
    args = parser.parse_args()

    import rootpy
    rootpy.log.basic_config_colorized()
    import logging
    if hasattr(logging, 'captureWarnings'):
        logging.captureWarnings(True)

    def formatwarning(message, category, filename, lineno, line=None):
        return "{0}: {1}".format(category.__name__, message)

    warnings.formatwarning = formatwarning
    args.ext = args.ext.strip('.')

    if args.quiet:
        warnings.simplefilter("ignore",
            RootNumpyUnconvertibleWarning)

    userfunc = None
    if args.script is not None:
        # get user-defined function
        try:
            exec(compile(open(args.script).read(), args.script, 'exec'),
                 globals(), locals())
        except IOError:
            sys.exit('Could not open script %s' % args.script)
        funcname = os.path.splitext(os.path.basename(args.script))[0]
        try:
            userfunc = locals()[funcname]
        except KeyError:
            sys.exit("Could not find the function '%s' in the script %s" %
                (funcname, args.script))

    for inputname in args.files:
        outputname = os.path.splitext(inputname)[0] + '.' + args.ext
        if os.path.exists(outputname) and not args.force:
            sys.exit(('Output %s already exists. '
                'Use the --force option to overwrite it') % outputname)
        try:
            rootfile = root_open(inputname)
        except IOError:
            sys.exit("Could not open %s" % inputname)
        try:
            if args.complevel > 0:
                filters = tables.Filters(complib=args.complib,
                                         complevel=args.complevel)
            else:
                filters = None
            hd5file = tables.openFile(filename=outputname, mode='w',
                                      title='Data', filters=filters)
        except IOError:
            sys.exit("Could not create %s" % outputname)
        try:
            log.info("Converting %s ..." % inputname)
            convert(rootfile, hd5file,
                    entries=args.entries,
                    userfunc=userfunc,
                    selection=args.selection)
            log.info("Created %s" % outputname)
        except KeyboardInterrupt:
            log.info("Caught Ctrl-c ... cleaning up")
            os.unlink(outputname)
            break
        finally:
            hd5file.close()
            rootfile.Close()
