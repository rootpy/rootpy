# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
"""
This module allows C++ template types to be generated on demand with ease,
automatically building dictionaries with ROOT's ACLiC as necessary.  Unlike
vanilla ACLiC, rootpy's stl module generates and compiles dictionaries without
creating a mess of temporary files in your current working directory.
Dictionaries are also cached in ``~/.cache/rootpy/`` and used by any future
request for the same dictionary instead of compiling from scratch again.
Templates can be arbitrarily nested, limited only by what ACLiC and CINT can
handle.

Examples
--------

.. sourcecode:: python

    import rootpy.stl as stl, ROOT

    # Create a vector type
    StrVector = stl.vector(stl.string)
    # Instantiate
    strvector = StrVector()
    strvector.push_back("Hello")

    MapStrRoot = stl.map(stl.string, ROOT.TH1D)
    MapStrRootPtr = stl.map(stl.string, "TH1D*")


Dictionary generation type inference is flexible and can be nested::

    >>> import rootpy.stl as stl
    >>> import ROOT
    >>> from rootpy.plotting import Hist
    >>> stl.vector('int')
    <class 'ROOT.vector<int,allocator<int> >'>
    >>> stl.vector(int)
    <class 'ROOT.vector<int,allocator<int> >'>
    >>> stl.vector(long)
    <class 'ROOT.vector<long,allocator<long> >'>
    >>> stl.vector('vector<int>')
    <class 'ROOT.vector<vector<int,allocator<int> >,allocator<vector<int,allocator<int> > > >'>
    >>> stl.vector(stl.vector('int'))
    <class 'ROOT.vector<vector<int,allocator<int> >,allocator<vector<int,allocator<int> > > >'>
    >>> stl.vector(stl.vector(stl.vector(int)))
    <class 'ROOT.vector<vector<vector<int,allocator<int> >,allocator<vector<int,allocator<int> > > > >'>
    >>> stl.map('string,int')
    <class 'ROOT.map<string,int,less<string>,allocator<pair<const string,int> > >'>
    >>> stl.map('string', 'int')
    <class 'ROOT.map<string,int,less<string>,allocator<pair<const string,int> > >'>
    >>> stl.map(stl.string, int)
    <class 'ROOT.map<string,int,less<string>,allocator<pair<const string,int> > >'>
    >>> stl.map(str, int)
    <class 'ROOT.map<string,int,less<string>,allocator<pair<const string,int> > >'>
    >>> stl.map(str, stl.map(int, stl.vector(float)))
    <class 'ROOT.map<string,map<int,vector<float,allocator<float> > > >'>
    >>> stl.map(str, Hist)
    <class 'ROOT.map<string,TH1,less<string>,allocator<pair<const string,TH1> > >'>
    >>> stl.map(str, ROOT.TH1)
    <class 'ROOT.map<string,TH1,less<string>,allocator<pair<const string,TH1> > >'>
    >>> stl.map(str, 'TH1*')
    <class 'ROOT.map<string,TH1*,less<string>,allocator<pair<const string,TH1*> > >'>

"""
from __future__ import absolute_import

import sys
import inspect
import hashlib
import os
import re
from os.path import join as pjoin, exists

import ROOT

from .extern.pyparsing import ParseException
from .extern.six import string_types

from .base import Object
from .defaults import extra_initialization
from .utils.cpp import CPPGrammar
from .utils.path import mkdir_p
from .utils.lock import lock
from . import compiled
from . import userdata
from . import lookup_by_name, register, QROOT
from . import log; log = log[__name__]

__all__ = []

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
    "pair": "<utility>",
    "string": "<string>",
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

NEW_DICTS = False
LOOKUP_TABLE_NAME = 'lookup'

# Initialized in initialize()
LOADED_DICTS = {}

DICTS_PATH = os.path.join(userdata.BINARY_PATH, 'dicts')
if not os.path.exists(DICTS_PATH):
    # avoid race condition by ignoring OSError if path exists by the time we
    # try to create it. See https://github.com/rootpy/rootpy/issues/328
    mkdir_p(DICTS_PATH)

include_list = os.path.join(userdata.BINARY_PATH, 'include_paths.list')
log.debug('Using {0} to get additional include paths'.format(include_list))
if os.path.exists(include_list):
    with open(include_list) as inc_list:
        for line in inc_list:
            line = line.strip()
            log.debug('adding {0} to the include paths'.format(line))
            ROOT.gInterpreter.AddIncludePath(line)

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

    def ensure_built(self, headers=None):
        """
        Make sure that a dictionary exists for this type.
        """
        if not self.params:
            return
        else:
            for child in self.params:
                child.ensure_built(headers=headers)
        if headers is None:
            headers = self.guess_headers
        generate(str(self), headers,
                 has_iterators=self.name in HAS_ITERATORS)

    @property
    def guess_headers(self):
        """
        Attempt to guess what headers may be required in order to use this
        type. Returns `guess_headers` of all children recursively.

        * If the typename is in the :const:`KNOWN_TYPES` dictionary, use the
            header specified there
        * If it's an STL type, include <{type}>
        * If it exists in the ROOT namespace and begins with T,
          include <{type}.h>
        """
        name = self.name.replace("*", "")
        headers = []
        if name in KNOWN_TYPES:
            headers.append(KNOWN_TYPES[name])
        elif name in STL:
            headers.append('<{0}>'.format(name))
        elif hasattr(ROOT, name) and name.startswith("T"):
            headers.append('<{0}.h>'.format(name))
        elif '::' in name:
            headers.append('<{0}.h>'.format(name.replace('::', '/')))
        elif name == 'allocator':
            headers.append('<memory>')
        else:
            try:
                # is this just a basic type?
                CPPGrammar.BASIC_TYPE.parseString(name, parseAll=True)
            except ParseException as e:
                # nope... I don't know what it is
                log.warning(
                    "unable to guess headers required for {0}".format(name))
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
    """
    if inspect.isclass(obj):
        if issubclass(obj, Object):
            return obj._ROOT.__name__
        if issubclass(obj, string_types):
            return 'string'
        return obj.__name__
    if not isinstance(obj, string_types):
        raise TypeError("expected string or class")
    return obj


def generate(declaration, headers=None, has_iterators=False):
    """Compile and load the reflection dictionary for a type.

    If the requested dictionary has already been cached, then load that instead.

    Parameters
    ----------
    declaration : str
        A type declaration (for example "vector<int>")
    headers : str or list of str
        A header file or list of header files required to compile the dictionary
        for this type.
    has_iterators : bool
        If True, then include iterators in the dictionary generation.
    """
    global NEW_DICTS
    # FIXME: _rootpy_dictionary_already_exists returns false positives
    # if a third-party module provides "incomplete" dictionaries.
    #if compiled._rootpy_dictionary_already_exists(declaration):
    #    log.debug("generate({0}) => already available".format(declaration))
    #    return
    log.debug("requesting dictionary for {0}".format(declaration))
    if headers:
        if isinstance(headers, string_types):
            headers = sorted(headers.split(';'))
        log.debug("using the headers {0}".format(', '.join(headers)))
        unique_name = ';'.join([declaration] + headers)
    else:
        unique_name = declaration
    unique_name = unique_name.replace(' ', '')

    # If the library is already loaded, do nothing
    if unique_name in LOADED_DICTS:
        log.debug("dictionary for {0} is already loaded".format(declaration))
        return

    if sys.version_info[0] < 3:
        libname = hashlib.sha512(unique_name).hexdigest()[:16]
    else:
        libname = hashlib.sha512(unique_name.encode('utf-8')).hexdigest()[:16]
    libnameso = libname + ".so"

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
        if (ROOT.gInterpreter.Load(pjoin(DICTS_PATH, libnameso))
                not in (0, 1)):
            raise RuntimeError(
                "failed to load the library for '{0}' @ {1}".format(
                    declaration, libname))
        LOADED_DICTS[unique_name] = None
        return

    with lock(pjoin(DICTS_PATH, "lock"), poll_interval=5, max_age=60):
        # This dict was not previously generated so we must create it now
        log.info("generating dictionary for {0} ...".format(declaration))
        includes = ''
        if headers is not None:
            for header in headers:
                if re.match('^<.+>$', header):
                    includes += '#include {0}\n'.format(header)
                else:
                    includes += '#include "{0}"\n'.format(header)
        source = LINKDEF % locals()
        sourcepath = os.path.join(DICTS_PATH, '{0}.C'.format(libname))
        log.debug("source path: {0}".format(sourcepath))
        with open(sourcepath, 'w') as sourcefile:
            sourcefile.write(source)
        log.debug("include path: {0}".format(
            ROOT.gSystem.GetIncludePath()))
        if (ROOT.gSystem.CompileMacro(
                sourcepath, 'k-', libname, DICTS_PATH) != 1):
            raise RuntimeError(
                "failed to compile the library for '{0}'".format(
                    sourcepath))

    LOADED_DICTS[unique_name] = None
    NEW_DICTS = True


Template = QROOT.Template


class SmartTemplate(Template):
    """
    Behaves like ROOT's Template class, except it will build dictionaries on
    demand.
    """
    def __call__(self, *params, **kwargs):
        """
        Instantiate the template represented by ``self`` with the template
        arguments specified by ``params``.
        """
        headers = kwargs.pop('headers', None)
        params = ", ".join(make_string(p) for p in params)
        typ = self.__name__
        if params:
            typ = '{0}<{1}>'.format(typ, params)
        cpptype = CPPType.from_string(typ)
        str_name = str(cpptype)
        # check registry
        cls = lookup_by_name(str_name)
        if cls is None:
            cpptype.ensure_built(headers=headers)
            cls = Template.__call__(self, params)
            register(names=str_name, builtin=True)(cls)
        return cls


from .utils.module_facade import Facade


@Facade(__name__, expose_internal=False)
class STLWrapper(object):
    # Base types
    for t in STL:
        locals()[t] = SmartTemplate(t)
    del t
    string = QROOT.string
    CPPType = CPPType
    generate = staticmethod(generate)
