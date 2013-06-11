# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
"""
:py:mod:`rootpy.stl`
====================

This module allows C++ template types to be generated on demand with ease,
automatically building dictionaries with :py:mod:`rootpy.rootcint` as necessary.

It contains a C++ template typename parser written in
:py:mod:`rootpy.extern.pyparsing`.

Example use:

.. sourcecode:: python

    import rootpy.stl as stl, ROOT

    # Create a vector type
    StrVector = stl.vector(stl.string)
    # Instantiate
    strvector = StrVector()
    strvector.push_back("Hello")
    # etc.

    MapStrRoot = stl.map(stl.string, ROOT.TH1D)
    MapStrRootPtr = stl.map(stl.string, "TH1D*")

"""
from __future__ import absolute_import

import hashlib
import os
import re
import subprocess

from os.path import join as pjoin, exists

import ROOT

from .extern.pyparsing import ParseException
from .extern.lockfile import LockFile

from .defaults import extra_initialization
from .util.cpp import CPPGrammar
from . import compiled
from . import userdata
from . import lookup_by_name, register, QROOT
from . import log; log = log[__name__]

STL = QROOT.std.stlclasses
HAS_ITERATORS = [
    'map',
    'vector',
    'list'
]
KNOWN_TYPES = {
    # Specify class names and headers to use here. ROOT classes beginning "T"
    # and having a header called {class}.h are picked up automatically.
    # 'TLorentzVector': 'TLorentzVector.h',
    "pair" : "utility",
}


# FIXME: _rootpy_dictionary_already_exists returns false positives
# if a third-party module provides "incomplete" dictionaries.
compiled.register_code("""
    #include <string>

    // PyROOT builtin
    namespace PyROOT { namespace Utility {
        const std::string ResolveTypedef( const std::string& name );
    } }

    // cint magic
    int G__defined_tagname(const char*, int);

    // Returns true if the given type does not require a dictionary
    bool _rootpy_dictionary_already_exists(const char* type) {
        const std::string full_typedef = PyROOT::Utility::ResolveTypedef(type);
        return G__defined_tagname(full_typedef.c_str(), 4) != -1;
    }
""", ["_rootpy_dictionary_already_exists"])

LINKDEF = '''\
%(includes)s
#ifdef __CINT__
#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;
#pragma link C++ nestedclasses;
#pragma link C++ nestedtypedefs;
#pragma link C++ class %(declaration)s;
#pragma link C++ class %(declaration)s::*;
#ifdef HAS_ITERATOR
#pragma link C++ operators %(declaration)s::iterator;
#pragma link C++ operators %(declaration)s::const_iterator;
#pragma link C++ operators %(declaration)s::reverse_iterator;
#pragma link C++ operators %(declaration)s::const_reverse_iterator;
#endif
#endif
'''

def root_config(*flags):

    flags = subprocess.Popen(
        ['root-config'] + list(flags),
        stdout=subprocess.PIPE).communicate()[0].strip().split()
    flags = ' '.join(['-I'+os.path.realpath(p[2:]) if
        p.startswith('-I') else p for p in flags])
    return flags


def shell(cmd):

    log.debug(cmd)
    return subprocess.call(cmd, shell=True)


ROOT_INC = root_config('--incdir')
ROOT_LDFLAGS = root_config('--libs', '--ldflags')
ROOT_CXXFLAGS = root_config('--cflags')
CXX = root_config('--cxx')
LD = root_config('--ld')

NEW_DICTS = False
LOOKUP_TABLE_NAME = 'lookup'
USE_ACLIC = True

# Initialized in initialize()
LOADED_DICTS = {}

DICTS_PATH = os.path.join(userdata.BINARY_PATH, 'dicts')
if not os.path.exists(DICTS_PATH):
    os.makedirs(DICTS_PATH)

@extra_initialization
def initialize():
    global DICTS_PATH

    # Used insetad of AddDynamicPath for ordering
    path = ":".join([DICTS_PATH, ROOT.gSystem.GetDynamicPath()])
    ROOT.gSystem.SetDynamicPath(path)

    ROOT.gSystem.AddLinkedLibs("-Wl,-rpath,{0}".format(DICTS_PATH))

class CPPType(CPPGrammar):
    """
    Grammar and representation of a C++ template type. Can handle arbitrary
    nesting and namespaces.
    """
    def __init__(self, parse_result):

        self.parse_result = parse_result
        self.prefix = parse_result.type_prefix
        self.name = ' '.join(parse_result.type_name)
        self.params = parse_result.template_params
        self.member = parse_result.template_member
        self.suffix = parse_result.type_suffix

    def __repr__(self):

        return self.parse_result.dump()

    @classmethod
    def make(cls, string, location, tokens):

        return cls(tokens)

    @property
    def is_template(self):
        """
        Is this a template type? (Does it have template parameters?)
        """
        return bool(self.params)

    def ensure_built(self):
        """
        Make sure that a dictionary exists for this type.
        """
        if not self.params:
            return
        else:
            for child in self.params:
                child.ensure_built()
        generate(str(self), self.guess_headers,
                has_iterators=self.name in HAS_ITERATORS)

    @property
    def guess_headers(self):
        """
        Attempt to guess what headers may be required in order to use this type.
        Returns `guess_headers` of all children recursively.

        * If the typename is in the :const:`KNOWN_TYPES` dictionary, use the
            header specified there
        * If it's an STL type, include <{type}>
        * If it exists in the ROOT namespace and begins with T, include <{type}.h>
        """
        name = self.name.replace("*", "")
        headers = []
        if name in KNOWN_TYPES:
            headers.append(KNOWN_TYPES[name])
        elif name in STL:
            headers.append('<%s>' % name)
        elif hasattr(ROOT, name) and name.startswith("T"):
            headers.append('<%s.h>' % name)
        if self.params:
            for child in self.params:
                headers.extend(child.guess_headers)
        # remove duplicates
        return list(set(headers))

    @property
    def cls(self):
        """
        Return the class definition for this type
        """
        # TODO: register the resulting type?
        return SmartTemplate(self.name)(", ".join(map(str, self.params)))

    @classmethod
    def try_parse(cls, string):
        """
        Try to parse ``string`` as a C++ type, returning :const:`None` on
        failure.
        """
        try:
            with log.ignore("^Failed to parse.*$"):
                return cls.from_string(string)
        except ParseException:
            return None

    @classmethod
    def from_string(cls, string):
        """
        Parse ``string`` into a CPPType instance
        """
        cls.TYPE.setParseAction(cls.make)
        try:
            return cls.TYPE.parseString(string, parseAll=True)[0]
        except ParseException:
            log.error("Failed to parse '{0}'".format(string))
            raise

    def __str__(self):
        """
        Returns the C++ code representation of this type
        """
        prefix = ' '.join(self.prefix)
        if prefix:
            prefix += ' '
        name = self.name
        args = [str(p) for p in self.params] if self.params else []
        templatize = '<{0} >' if args and args[-1].endswith('>') else '<{0}>'
        args = '' if not self.params else templatize.format(', '.join(args))
        member = ('::' + self.member[0]) if self.member else ''
        suffix = ' '.join(self.suffix)
        return "{0}{1}{2}{3}{4}".format(prefix, name, args, member, suffix)


def make_string(obj):
    """
    If ``obj`` is a string, return that, otherwise attempt to figure out the
    name of a type.

    Example:

    .. sourcecode:: python

        make_string(ROOT.TH1D) == "TH1D")
    """
    if not isinstance(obj, basestring):
        if hasattr(obj, "__name__"):
            obj = obj.__name__
        else:
            raise RuntimeError("Expected string or class")
    return obj


def generate(declaration,
        headers=None, has_iterators=False):
    global NEW_DICTS

    # FIXME: _rootpy_dictionary_already_exists returns false positives
    # if a third-party module provides "incomplete" dictionaries.
    #if compiled._rootpy_dictionary_already_exists(declaration):
    #    log.debug("generate({0}) => already available".format(declaration))
    #    return

    log.debug("requesting dictionary for %s" % declaration)
    if headers:
        if isinstance(headers, basestring):
            headers = sorted(headers.split(';'))
        unique_name = ';'.join([declaration] + headers)
    else:
        unique_name = declaration
    unique_name = unique_name.replace(' ', '')
    # The library is already loaded, do nothing
    if unique_name in LOADED_DICTS:
        log.debug("dictionary for {0} is already loaded".format(declaration))
        return

    libname = hashlib.sha512(unique_name).hexdigest()[:16]
    libnameso = libname + ".so"

    with LockFile(pjoin(DICTS_PATH, "lock")):

        if ROOT.gROOT.GetVersionInt() < 53403:
            # check for this class in the global TClass list and remove it
            # fixes infinite recursion in ROOT < 5.34.03
            # (exact ROOT versions where this is required is unknown)
            cls = ROOT.gROOT.GetClass(declaration)
            if cls and not cls.IsLoaded():
                log.debug("removing {0} from gROOT.GetListOfClasses()".format(
                    declaration))
                ROOT.gROOT.GetListOfClasses().Remove(cls)

        # If a .so already exists for this class, use it.
        if exists(pjoin(DICTS_PATH, libnameso)):
            log.debug("loading previously generated dictionary for {0}"
                      .format(declaration))
            if ROOT.gInterpreter.Load(pjoin(DICTS_PATH, libnameso)) not in (0, 1):
                raise RuntimeError("failed to load the library for '{0}' @ {1}"
                    .format(declaration, libname))
            LOADED_DICTS[unique_name] = None
            return

        # This dict was not previously generated so we must create it now
        log.info("generating dictionary for {0} ...".format(declaration))
        includes = ''
        if headers is not None:
            for header in headers:
                if re.match('^<.+>$', header):
                    includes += '#include %s\n' % header
                else:
                    includes += '#include "%s"\n' % header
        source = LINKDEF % locals()
        if USE_ACLIC:
            sourcepath = os.path.join(DICTS_PATH, '{0}.C'.format(libname))
            log.debug("source path: {0}".format(sourcepath))
            with open(sourcepath, 'w') as sourcefile:
                sourcefile.write(source)

            if ROOT.gSystem.CompileMacro(sourcepath, 'k-', libname, DICTS_PATH) != 1:
                raise RuntimeError("failed to compile the library for '{0}'".format(sourcepath))
        else:
            cwd = os.getcwd()
            os.chdir(DICTS_PATH)
            sourcepath = os.path.join(DICTS_PATH, 'LinkDef.h')
            OPTS_FLAGS = ''
            if has_iterators:
                OPTS_FLAGS = '-DHAS_ITERATOR'
            all_vars = dict(globals(), **locals())
            with open(sourcepath, 'w') as sourcefile:
                sourcefile.write(source)
            # run rootcint
            if shell('rootcint -f dict.cxx -c -p {OPTS_FLAGS} '
                     '-I{ROOT_INC} LinkDef.h'.format(**all_vars)):
                os.chdir(cwd)
                raise RuntimeError('rootcint failed for %s' % declaration)
            # add missing includes
            os.rename('dict.cxx', 'dict.tmp')
            with open('dict.cxx', 'w') as patched_source:
                patched_source.write(includes)
                with open('dict.tmp', 'r') as orig_source:
                    patched_source.write(orig_source.read())
            if shell('{CXX} {ROOT_CXXFLAGS} {OPTS_FLAGS} '
                     '-Wall -fPIC -c dict.cxx -o dict.o'.format(**all_vars)):
                os.chdir(cwd)
                raise RuntimeError('failed to compile %s' % declaration)
            if shell('{LD} {ROOT_LDFLAGS} -Wall -shared '
                   'dict.o -o {libname}.so'.format(**all_vars)):
                os.chdir(cwd)
                raise RuntimeError('failed to link %s' % declaration)
            # load the newly compiled library
            if ROOT.gInterpreter.Load(pjoin(DICTS_PATH, libnameso)) not in (0, 1):
                os.chdir(cwd)
                raise RuntimeError('failed to load the library for %s' % declaration)
            os.chdir(cwd)

    LOADED_DICTS[unique_name] = None
    NEW_DICTS = True


Template = QROOT.Template

class SmartTemplate(Template):
    """
    Behaves like ROOT's Template class, except it will build dictionaries on
    demand.
    """
    def __call__(self, *args):
        """
        Instantiate the template represented by ``self`` with the template
        arguments specified by ``args``.
        """
        params = ", ".join(make_string(p) for p in args)

        typ = self.__name__
        if params:
            typ = '{0}<{1}>'.format(typ, params)
        cpptype = CPPType.from_string(typ)
        str_name = str(cpptype)
        # check registry
        cls = lookup_by_name(str_name)
        if cls is None:
            cpptype.ensure_built()
            cls = Template.__call__(self, params)
            register(names=str_name, builtin=True)(cls)
        return cls


from rootpy.extern.module_facade import Facade

@Facade(__name__, expose_internal=False)
class STLWrapper(object):
    # Base types
    for t in STL:
        locals()[t] = SmartTemplate(t)
    del t
    string = QROOT.string
    CPPType = CPPType
    generate = staticmethod(generate)
