# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
from __future__ import absolute_import

from urllib2 import urlopen
import xml.dom.minidom as minidom
from itertools import chain, izip

from .. import log; log = log[__name__]
from . import quickroot as QROOT

__all__ = [
    'iter_ROOT_classes',
    'humanize_bytes',
    'print_table',
    'izip_exact',
    'LengthMismatch',
]


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



class LengthMismatch(Exception):
    pass


def _throw():
    raise LengthMismatch
    yield None # unreachable


def _check(rest):
    for i in rest:
        try:
            i.next()
        except LengthMismatch:
            pass
        else:
            raise LengthMismatch
    return
    yield None # unreachable


def izip_exact(*iterables):
    """
    A lazy izip() that ensures that all iterables have the same length.
    A LengthMismatch exception is raised if the iterables' lengths differ.

    Examples
    --------

        >>> list(zip_exc([]))
        []
        >>> list(zip_exc((), (), ()))
        []
        >>> list(zip_exc("abc", range(3)))
        [('a', 0), ('b', 1), ('c', 2)]
        >>> try:
        ...     list(zip_exc("", range(3)))
        ... except LengthMismatch:
        ...     print "mismatch"
        mismatch
        >>> try:
        ...     list(zip_exc(range(3), ()))
        ... except LengthMismatch:
        ...     print "mismatch"
        mismatch
        >>> try:
        ...     list(zip_exc(range(3), range(2), range(4)))
        ... except LengthMismatch:
        ...     print "mismatch"
        mismatch
        >>> items = zip_exc(range(3), range(2), range(4))
        >>> items.next()
        (0, 0, 0)
        >>> items.next()
        (1, 1, 1)
        >>> try: items.next()
        ... except LengthMismatch: print "mismatch"
        mismatch

    References
    ----------

    [1] http://code.activestate.com/recipes/497006-zip_exc-a-lazy-zip-that-ensures-that-all-iterables/

    """
    rest = [chain(i, _throw()) for i in iterables[1:]]
    first = chain(iterables[0], _check(rest))
    return izip(*[first] + rest)
