from __future__ import print_function
import sys
import ROOT
from ROOT import MethodProxy
import inspect
import urllib2, xml.dom.minidom as minidom
from rootpy.util.cpp import CPPGrammar
from nose.plugins.attrib import attr


@attr('slow')
def test_cpp():

    def get_all_ROOT_classes():

        class_names = [s.childNodes[0].nodeValue
            for s in minidom.parse(
                urllib2.urlopen("http://root.cern.ch/root/html/ClassIndex.html")
                    ).getElementsByTagName("span")
                        if ("class", "typename") in s.attributes.items()]
        classes = []
        for name in class_names:
            try:
                classes.append(getattr(ROOT, name))
            except AttributeError:
                pass
        print('%d ROOT classes' % len(classes))
        return classes

    i = 0
    num_methods = 0

    for cls in get_all_ROOT_classes():
        members = inspect.getmembers(cls)
        # filter out those starting with "_" or "operator "
        # and non-method members
        # also split overloaded methods
        methods = {}
        for name, func in members:
            if name.startswith('_') or name.startswith('operator'):
                continue
            if not isinstance(func, MethodProxy):
                continue
            methods[name] = (func, func.func_doc.split('\n'))

        for name, (func, sigs) in methods.items():
            for sig in sigs:
                num_methods += 1
                if CPPGrammar.parse_method(sig, silent=False):
                    i += 1
            print("{0} / {1}".format(i, num_methods), end='\r')
            sys.stdout.flush()
    print("{0} / {1}".format(i, num_methods))

if __name__ == "__main__":
    import nose
    nose.runmodule()
