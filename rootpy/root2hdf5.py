# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
"""
This module handles conversion of ROOT's TFile and
contained TTrees into HDF5 format with PyTables
"""
import os
import sys
import tables
import warnings

from .io import open as ropen, utils
from . import log; log = log[__name__]
from .extern.progressbar import ProgressBar, Bar, ETA, Percentage
from .logger.util import check_tty

from root_numpy import tree2rec, RootNumpyUnconvertibleWarning


def convert(rfile, hfile, rpath='', entries=-1):

    isatty = check_tty(sys.stdout)
    if isatty:
        widgets = [Percentage(), ' ', Bar(), ' ', ETA()]

    if isinstance(hfile, basestring):
        hfile = tables.openFile(filename=hfile, mode="w", title="Data")
    if isinstance(rfile, basestring):
        rfile = ropen(rfile)

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

        for tree, treename in [
                (rfile.Get(os.path.join(dirpath, treename)), treename)
                for treename in treenames]:

            log.info("Converting tree '%s' with %i entries ..." % (treename,
                tree.GetEntries()))

            total_entries = tree.GetEntries()
            if isatty:
                pbar = ProgressBar(widgets=widgets, maxval=total_entries)

            if entries <= 0:
                # read the entire tree
                if isatty:
                    pbar.start()
                recarray = tree2rec(tree)
                table = hfile.createTable(
                    group, treename, recarray, tree.GetTitle())
                table.flush()
            else:
                # read the tree in chunks
                offset = 0
                while offset < total_entries:
                    if offset > 0:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore",
                                    RootNumpyUnconvertibleWarning)
                            recarray = tree2rec(tree,
                                    entries=entries, offset=offset)
                        table.append(recarray)
                    else:
                        recarray = tree2rec(tree,
                                entries=entries, offset=offset)
                        if isatty:
                            # start after any output from root_numpy
                            pbar.start()
                        table = hfile.createTable(
                            group, treename, recarray, tree.GetTitle())
                    offset += entries
                    if isatty and offset <= total_entries:
                        pbar.update(offset)
                    table.flush()
            if isatty:
                pbar.finish()


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

    for inputname in args.files:
        outputname = os.path.splitext(inputname)[0] + '.' + args.ext
        if os.path.exists(outputname) and not args.force:
            sys.exit(('Output %s already exists. '
                'Use the --force option to overwrite it') % outputname)
        try:
            rootfile = ropen(inputname)
        except IOError:
            sys.exit("Could not open %s" % inputname)
        try:
            filters = tables.Filters(
                    complib=args.complib, complevel=args.complevel)
            hd5file = tables.openFile(filename=outputname, mode='w',
                    title='Data', filters=filters)
        except IOError:
            sys.exit("Could not create %s" % outputname)
        try:
            log.info("Converting %s ..." % inputname)
            convert(rootfile, hd5file, entries=args.entries)
            log.info("Created %s" % outputname)
        except KeyboardInterrupt:
            log.info("Caught Ctrl-c ... cleaning up")
            os.unlink(outputname)
            break
        finally:
            hd5file.close()
            rootfile.Close()
