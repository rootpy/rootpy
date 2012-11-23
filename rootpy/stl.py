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

import re
import sys

import ROOT

from ROOT import Template


from rootpy.extern.pyparsing import (Combine, Forward, Group, Literal, Optional,
    Word, ZeroOrMore, alphanums, delimitedList, stringStart, stringEnd, ungroup,
    ParseException)
    
from rootpy.rootcint import generate
from . import log; log = log[__name__]

STL = ROOT.std.stlclasses
KNOWN_TYPES = {
    # Specify class names and headers to use here. ROOT classes beginning "T"
    # and having a header called {class}.h are picked up automatically.
    # 'TLorentzVector': 'TLorentzVector.h',
    "pair" : "utility",
}

class ParsedObject(object):
    def __repr__(self):
        return "<%s %s>" % (self.__class__.__name__, self.tokens)
        
    @classmethod
    def make(cls, string, location, tokens):
        result = cls(tokens.asList())
        result.expression = string
        return result
        
class CPPType(ParsedObject):
    """
    Grammar and representation of a C++ template type. Can handle arbitrary
    nesting and namespaces.
    """
    PTR = ZeroOrMore(Word("*") | Word("&"))
    NAME = Combine(Word(alphanums) + PTR).setName("C++ name")("token")
    NS_SEPARATOR = Literal("::").setName("Namespace Separator")
    NAMESPACED_NAME = Combine(NAME + ZeroOrMore(NS_SEPARATOR + NAME))

    TYPE = Forward()

    TEMPLATE_PARAMS = Optional(
        Literal("<").suppress()
        + Group(delimitedList(TYPE))("params")
        + Literal(">").suppress(), default=None)
        
    CLASS_MEMBER = Optional(
        NS_SEPARATOR.suppress()
        + NAMESPACED_NAME.setName("class member"), default=None
    )("class_member")

    TYPE << NAMESPACED_NAME + TEMPLATE_PARAMS + CLASS_MEMBER
    
    def __init__(self, tokens):
        # This line mirrors the "TYPE << ..." definition above.
        self.name, self.params, self.member = self.tokens = tokens
    
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
        generate(str(self), self.guess_headers)
    
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
        return headers
            
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
        name = self.name
        args = [str(p) for p in self.params] if self.params else []
        templatize = "<{0} >" if args and args[-1].endswith(">") else "<{0}>"
        args = "" if not self.params else templatize.format(", ".join(args))
        member = ("::"+self.member) if self.member else ""
        return "{0}{1}{2}".format(name, args, member)

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
        cpptype.ensure_built()
        
        log.debug("Building type {0}".format(typ))
        # TODO: Register the type?
        return Template.__call__(self, params)

from rootpy.extern.module_facade import Facade

@Facade(__name__, expose_internal=False)
class STLWrapper(object):
    # Base types
    for t in STL:
        locals()[t] = SmartTemplate(t)
    del t
    string = ROOT.string

    CPPType = CPPType

