# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
from __future__ import absolute_import

import re

from .cut import Cut

__all__ = [
    'Categories',
]


class Categories(object):
    """
    Implements a mechanism to ease the creation of cuts that describe
    non-overlapping categories.
    """
    #TODO: use pyparsing
    CUT_REGEX = '[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?'
    NODE_PATTERN = re.compile(
        '^{(?P<variable>[^:|]+)(?::(?P<type>[IFif]))?\|'
        '(?P<leftchild>{.+})?(?P<cut>' + CUT_REGEX + ')'
        '(?P<rightchild>{.+})?}$')
    CATEGORY_PATTERN = re.compile(
        '^(?P<left>{.+})(?:x(?P<right>{.+}(?:x{.+})*))$')
    CATEGORY_NODE_PATTERN = re.compile(
        '^{(?P<variable>[^:|]+)(?::(?P<type>[IFif]))?\|'
        '(?P<cuts>[\*]?(?:' + CUT_REGEX + ')(?:,' + CUT_REGEX + ')*[\*]?)}$')

    @classmethod
    def from_string(cls, string, variables=None):
        node = None
        if variables is None:
            variables = []
        nodematch = re.match(Categories.NODE_PATTERN, string)
        categorymatch = re.match(Categories.CATEGORY_PATTERN, string)
        categorynodematch = re.match(Categories.CATEGORY_NODE_PATTERN, string)
        if categorymatch:
            node = cls.from_string(categorymatch.group('left'), variables)
            subtree = cls.from_string(categorymatch.group('right'), variables)
            incompletenodes = node.get_incomplete_children()
            for child in incompletenodes:
                if not child.leftchild and not child.forbidleft:
                    clone = subtree.clone()
                    child.set_left(clone)
                if not child.rightchild and not child.forbidright:
                    clone = subtree.clone()
                    child.set_right(clone)
        elif categorynodematch:
            var_type = 'F'
            if categorynodematch.group('type'):
                var_type = categorynodematch.group('type').upper()
            variable = (categorynodematch.group('variable'), var_type)
            if variable not in variables:
                variables.append(variable)
            cuts = categorynodematch.group('cuts').split(',')
            if len(cuts) != len(set(cuts)):
                raise SyntaxError(
                    "repeated cuts in '{0}'".format(
                        categorynodematch.group('cuts')))
            if sorted(cuts) != cuts:
                raise SyntaxError(
                    "cuts not in ascending order in '{0}'".format(
                        categorynodematch.group('cuts')))
            nodes = []
            for cut in cuts:
                actual_cut = cut.replace('*', '')
                node = Categories(
                    feature=variables.index(variable),
                    data=actual_cut,
                    variables=variables)
                if cut.startswith('*'):
                    node.forbidleft = True
                if cut.endswith('*'):
                    node.forbidright = True
                nodes.append(node)
            node = Categories.make_balanced_tree(nodes)
        elif nodematch:
            var_type = 'F'
            if nodematch.group('type'):
                var_type = nodematch.group('type').upper()
            variable = (nodematch.group('variable'), var_type)
            if variable not in variables:
                variables.append(variable)
            node = Categories(
                feature=variables.index(variable),
                data=nodematch.group('cut'),
                variables=variables)
            if nodematch.group('leftchild'):
                leftchild = cls.from_string(
                    nodematch.group('leftchild'), variables)
                node.set_left(leftchild)
            if nodematch.group('rightchild'):
                rightchild = cls.from_string(
                    nodematch.group('rightchild'), variables)
                node.set_right(rightchild)
        else:
            raise SyntaxError(
                "{0} is not valid category tree syntax".format(string))
        return node

    @classmethod
    def make_balanced_tree(cls, nodes):
        if len(nodes) == 0:
            return None
        if len(nodes) == 1:
            return nodes[0]
        center = len(nodes) // 2
        leftnodes = nodes[:center]
        rightnodes = nodes[center + 1:]
        node = nodes[center]
        leftchild = Categories.make_balanced_tree(leftnodes)
        rightchild = Categories.make_balanced_tree(rightnodes)
        node.set_left(leftchild)
        node.set_right(rightchild)
        return node

    def __init__(self,
                 feature,
                 data,
                 variables,
                 leftchild=None,
                 rightchild=None,
                 parent=None,
                 forbidleft=False,
                 forbidright=False):
        self.feature = feature
        self.data = data
        self.variables = variables
        self.leftchild = leftchild
        self.rightchild = rightchild
        self.parent = parent
        self.forbidleft = forbidleft
        self.forbidright = forbidright

    def clone(self):
        leftclone = None
        if self.leftchild is not None:
            leftclone = self.leftchild.clone()
        rightclone = None
        if self.rightchild is not None:
            rightclone = self.rightchild.clone()
        return Categories(
            self.feature,
            self.data,
            self.variables,
            leftclone,
            rightclone,
            self.parent,
            self.forbidleft,
            self.forbidright)

    def __str__(self):
        leftstr = ''
        rightstr = ''
        if self.forbidleft:
            leftstr = '*'
        elif self.leftchild is not None:
            leftstr = str(self.leftchild)
        if self.forbidright:
            rightstr = '*'
        elif self.rightchild is not None:
            rightstr = str(self.rightchild)
        if self.feature >= 0:
            return '{{0}:{1}|{2}{3}{4}}'.format(
                self.variables[self.feature],
                leftstr, str(self.data), rightstr)
        return '{<<leaf>>|{0}}'.format(str(self.data))

    def __repr__(self):
        return self.__str__()

    def set_left(self, child):
        if child is self:
            raise ValueError("attempted to set self as left child!")
        self.leftchild = child
        if child is not None:
            child.parent = self

    def set_right(self, child):
        if child is self:
            raise ValueError("attempted to set self as right child!")
        self.rightchild = child
        if child is not None:
            child.parent = self

    def is_leaf(self):
        return self.leftchild is None and self.rightchild is None

    def is_complete(self):
        return self.leftchild is not None and self.rightchild is not None

    def depth(self):
        leftdepth = 0
        if self.leftchild is not None:
            leftdepth = self.leftchild.depth() + 1
        rightdepth = 0
        if self.rightchild is not None:
            rightdepth = self.rightchild.depth() + 1
        return max(leftdepth, rightdepth)

    def balance(self):
        leftdepth = 0
        rightdepth = 0
        if self.leftchild is not None:
            leftdepth = self.leftchild.depth() + 1
        if self.rightchild is not None:
            rightdepth = self.rightchild.depth() + 1
        return rightdepth - leftdepth

    def get_leaves(self):
        if self.is_leaf():
            return [self]
        leftleaves = []
        if self.leftchild is not None:
            leftleaves = self.leftchild.get_leaves()
        rightleaves = []
        if self.rightchild is not None:
            rightleaves = self.rightchild.get_leaves()
        return leftleaves + rightleaves

    def get_incomplete_children(self):
        children = []
        if not self.is_complete():
            children.append(self)
        if self.leftchild is not None:
            children += self.leftchild.get_incomplete_children()
        if self.rightchild is not None:
            children += self.rightchild.get_incomplete_children()
        return children

    def __len__(self):
        """
        Number of categories beneath current node
        """
        if self.is_leaf():
            total = 0
            if not self.forbidleft:
                total += 1
            if not self.forbidright:
                total += 1
            return total
        total = 0
        if not self.forbidleft and self.leftchild is not None:
            total += len(self.leftchild)
        if not self.forbidright and self.rightchild is not None:
            total += len(self.rightchild)
        return total

    def walk(self, expression=None):
        if expression is None:
            expression = Cut()
        if self.feature < 0:
            if expression:
                yield expression
        if not self.forbidleft:
            leftcondition = expression & Cut(
                '{0}<={1}'.format(
                    self.variables[self.feature][0], self.data))
            if self.leftchild is not None:
                for condition in self.leftchild.walk(leftcondition):
                    yield condition
            else:
                yield leftcondition
        if not self.forbidright:
            rightcondition = expression & Cut(
                '{0}>{1}'.format(
                    self.variables[self.feature][0], self.data))
            if self.rightchild is not None:
                for condition in self.rightchild.walk(rightcondition):
                    yield condition
            else:
                yield rightcondition

    def __iter__(self):
        """
        Iterator over leaf conditions
        """
        for category in self.walk():
            yield category
