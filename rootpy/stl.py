import ROOT
from ROOT import Template
from rootpy.rootcint import generate
import re
import sys


TEMPLATE_REGEX = re.compile('^(?P<type>[^,<]+)<(?P<params>.+)>$')
STL = ROOT.std.stlclasses
KNOWN_TYPES = {
    'TLorentzVector': 'TLorentzVector.h',
}


class TemplateNode(object):

    def __init__(self, name):

        self.name = name
        self.children = []

    def compile(self, verbose=False):
        """
        Recursively compile chilren
        """
        if not self.children:
            return
        for child in self.children:
            child.compile(verbose=verbose)
        generate(str(self), self.headers, verbose=verbose)

    @property
    def cls(self):

        return SmartTemplate(self.name)(','.join(map(str, self.children)))

    @property
    def headers(self):
        """
        Build list of required headers recursively
        """
        headers = []
        if self.name in STL:
            headers.append('<%s>' % self.name)
        elif self.name in KNOWN_TYPES:
            headers.append(KNOWN_TYPES[self.name])
        for child in self.children:
            headers.extend(child.headers)
        return headers

    def __repr__(self):

        return str(self)

    def __str__(self):

        if self.children and len(self.children[-1].children) == 0:
            return '%s<%s>' % (self.name, ','.join(map(str, self.children)))
        elif self.children:
            return '%s<%s >' % (self.name, ','.join(map(str, self.children)))
        else:
            return self.name


def parse_template(decl, parent=None):
    # build a template tree
    # compile recursively
    """
    if parent is None then declaration must be of the top-level
    Foo<A> form, otherwise it may be of the form Foo<A>,Bar<B>,...
    """
    if parent is None:
        # remove whitespace
        decl = decl.replace(' ', '')
        # repeated < or ,
        if re.search('(<){2}', decl) or re.search('(,){2}', decl):
            raise SyntaxError('not a valid template: %s' % decl)
    # parse Foo<A>
    match = re.match(TEMPLATE_REGEX, decl)
    if not parent:
        if not match:
            raise SyntaxError('not a valid template: %s' % decl)
        groups = match.groupdict()
        name = groups['type']
        params = groups['params']
        node = TemplateNode(name)
        parse_template(params, node)
        return node
    elif match and ',' not in decl:
        groups = match.groupdict()
        name = groups['type']
        params = groups['params']
        node = TemplateNode(name)
        parent.children.append(node)
        parse_template(params, node)
        return
    # parse basic types
    if not re.search('(<)|(>)', decl):
        parent.children.append(TemplateNode(decl))
        return
    # parse Foo<A>,Bar<B>,...
    # move from left to right and find first < and matching >
    # end of string or comma must follow, repeat
    param_bounds = []
    nbrac = 0
    intemplate = False
    left = 0
    right = -1
    for i, s, in enumerate(decl):
        if s == '>' and not intemplate:
            raise SyntaxError('not a valid template: %s' % decl)
        if intemplate and i == len(decl) - 1 and s != '>':
            # early termination
            raise SyntaxError('not a valid template: %s' % decl)
        if not intemplate and (s == ',' or i == len(decl) - 1):
            if s == ',':
                right = i
            else:
                right = i + 1
            if left != right:
                param_bounds.append((left, right))
            left = right + 1
            continue
        if s == '<':
            nbrac += 1
            intemplate = True
        elif s == '>':
            if not intemplate:
                raise SyntaxError('not a valid template: %s' % decl)
            nbrac -= 1
        if intemplate and nbrac == 0:
            # found the matching >
            right = i + 1
            param_bounds.append((left, right))
            left = right
            intemplate = False
    if len(param_bounds) == 1:
        bounds = param_bounds[0]
        if bounds[0] != 0 or bounds[1] != len(decl) or not match:
            raise SyntaxError('not a valid template: %s' % decl)
        groups = match.groupdict()
        name = groups['type']
        params = groups['params']
        node = TemplateNode(name)
        parent.children.append(node)
        parse_template(params, node)
        return
    for bounds in param_bounds:
        parse_template(decl[bounds[0]:bounds[1]], parent)


class SmartTemplate(Template):

    def __call__(self, params, verbose=False):

        template_tree = parse_template(
            '%s<%s >' % (self.__name__, params))
        template_tree.compile(verbose=verbose)
        return Template.__call__(self, params)

from rootpy.extern.module_facade import Facade

@Facade(__name__, expose_internal=False)
class STLWrapper(object):
    for t in STL:
        locals()[t] = SmartTemplate(t)
    del t
    string = ROOT.string
