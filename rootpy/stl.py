import ROOT
from ROOT import Template
from rootpy.classfactory import generate
import re


TEMPLATE_REGEX = re.compile('^(?P<type>[^,<]+)<(?P<params>.+)>$')

STL = ROOT.std.stlclasses
KNOWN_TYPES = {
    'TLorentzVector': 'TLorentzVector.h',
}


class TemplateNode(object):

    def __init__(self, name):

        self.name = name
        self.children = []

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

    def __init__(self, name, headers=None, includes=None):

        Template.__init__(self, name)
        self.headers = headers
        self.includes = includes

    def __call__(self, *args):

        headers = []
        print args
        for arg in args:
            print arg
            template_params_match = re.match(TEMPLATE_REGEX, arg)
            if template_params_match:
                groups = template_params_match.groupdict()
                cls = groups['type'].strip()
                params = groups['params'].strip()
                SmartTemplate(cls)(params)
                if cls in STL:
                    headers.append('<%s>' % cls)
                elif cls in KNOWN_TYPES:
                    headers.append(KNOWN_TYPES[cls])
            elif arg in KNOWN_TYPES:
                headers.append(KNOWN_TYPES[arg])
        if self.__name__ in STL:
            headers.append('<%s>' % self.__name__)
        elif self.__name__ in KNOWN_TYPES:
            headers.append(KNOWN_TYPES[self.__name__])
        if self.headers is not None:
            headers.extend(self.headers)
        headers = list(set(headers))
        print headers
        generate('%s<%s >' % (self.__name__, ','.join(args)), headers)
        return Template.__call__(self, *args)
