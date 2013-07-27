# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
from .. import log; log = log[__name__]
from . import quickroot as QROOT
from urllib2 import urlopen
import xml.dom.minidom as minidom


def iter_ROOT_classes():
    """
    Iterator over all available ROOT classes
    """
    class_index = "http://root.cern.ch/root/html/ClassIndex.html"
    for s in minidom.parse(urlopen(class_index)).getElementsByTagName("span"):
        if ("class", "typename") in s.attributes.items():
            class_name = s.childNodes[0].nodeValue
            try:
                yield getattr(QROOT, class_name)
            except AttributeError:
                pass


def humanize_bytes(bytes, precision=1):

    abbrevs = (
        (1<<50L, 'PB'),
        (1<<40L, 'TB'),
        (1<<30L, 'GB'),
        (1<<20L, 'MB'),
        (1<<10L, 'kB'),
        (1, 'bytes')
    )
    if bytes == 1:
        return '1 byte'
    for factor, suffix in abbrevs:
        if bytes >= factor:
            break
    return '%.*f %s' % (precision, bytes / factor, suffix)


def print_table(table, sep='  '):

    # Reorganize data by columns
    cols = zip(*table)
    # Compute column widths by taking maximum length of values per column
    col_widths = [ max(len(value) for value in col) for col in cols ]
    # Create a suitable format string
    format = sep.join(['%%-%ds' % width for width in col_widths ])
    # Print each row using the computed format
    for row in table:
        print format % tuple(row)
