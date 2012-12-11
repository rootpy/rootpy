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
