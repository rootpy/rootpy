import re
from .cut import Cut


nodepattern = re.compile(
    '^{(?P<variable>[^:|]+)(?::(?P<type>[IFif]))?\|'
    '(?P<leftchild>{.+})?(?P<cut>[0-9.]+)(?P<rightchild>{.+})?}$')
categorypattern = re.compile(
    '^(?P<left>{.+})(?:x(?P<right>{.+}(?:x{.+})*))$')
categorynodepattern = re.compile(
    '^{(?P<variable>[^:|]+)(?::(?P<type>[IFif]))?\|'
    '(?P<cuts>[*]?(?:[0-9.]+)(?:,[0-9.]+)*[*]?)}$')


def parse_tree(string, variables=None):

    node = None
    if variables == None:
        variables = []
    nodematch = re.match(nodepattern, string)
    categorymatch = re.match(categorypattern, string)
    categorynodematch = re.match(categorynodepattern, string)
    if categorymatch:
        node = parse_tree(categorymatch.group("left"), variables)
        subtree = parse_tree(categorymatch.group("right"), variables)
        incompletenodes = node.get_incomplete_children()
        for child in incompletenodes:
            if not child.leftchild:
                clone = subtree.clone()
                child.set_left(clone)
            if not child.rightchild:
                clone = subtree.clone()
                child.set_right(clone)
    elif categorynodematch:
        varType = 'F'
        if categorynodematch.group('type'):
            varType = categorynodematch.group('type').upper()
        variable = (categorynodematch.group('variable'), varType)
        if variable not in variables:
            variables.append(variable)
        cuts = categorynodematch.group('cuts').split(',')
        if len(cuts) != len(set(cuts)):
            raise SyntaxError(
                    "repeated cuts in '%s'" %
                    categorynodematch.group('cuts'))
        if sorted(cuts) != cuts:
            raise SyntaxError(
                    "cuts not in ascending order in '%s'" %
                    categorynodematch.group('cuts'))
        nodes = []
        for cut in cuts:
            actual_cut = cut.replace('*', '')
            node = Node(feature=variables.index(variable),
                        data=actual_cut,
                        variables=variables)
            if cut.startswith('*'):
                node.forbidleft = True
            if cut.endswith('*'):
                node.forbidright = True
            nodes.append(node)
        node = _make_balanced_tree(nodes)
    elif nodematch:
        varType = 'F'
        if nodematch.group('type'):
            varType = nodematch.group('type').upper()
        variable = (nodematch.group('variable'), varType)
        if variable not in variables:
            variables.append(variable)
        node = Node(feature=variables.index(variable),
                    data=nodematch.group('cut'),
                    variables=variables)
        if nodematch.group('leftchild'):
            leftchild = parse_tree(nodematch.group('leftchild'), variables)
            node.set_left(leftchild)
        if nodematch.group('rightchild'):
            rightchild = parse_tree(nodematch.group('rightchild'), variables)
            node.set_right(rightchild)
    else:
        raise SyntaxError("%s is not valid decision tree syntax" % string)
    return node


def _make_balanced_tree(nodes):

    if len(nodes) == 0:
        return None
    if len(nodes) == 1:
        return nodes[0]
    center = len(nodes) / 2
    leftnodes = nodes[:center]
    rightnodes = nodes[center + 1:]
    node = nodes[center]
    leftchild = _make_balanced_tree(leftnodes)
    rightchild = _make_balanced_tree(rightnodes)
    node.set_left(leftchild)
    node.set_right(rightchild)
    return node


class Node(object):

    def __init__(self,
                 feature,
                 data,
                 variables,
                 leftchild=None,
                 rightchild=None,
                 parent=None):

        self.feature = feature
        self.data = data
        self.variables = variables
        self.leftchild = leftchild
        self.rightchild = rightchild
        self.parent = parent
        self.forbidleft = False
        self.forbidright = False

    def clone(self):

        leftclone = None
        if self.leftchild != None:
            leftclone = self.leftchild.clone()
        rightclone = None
        if self.rightchild != None:
            rightclone = self.rightchild.clone()
        return Node(self.feature,
                self.data,
                self.variables,
                leftclone,
                rightclone,
                self.parent)

    def __str__(self):

        leftstr = ''
        rightstr = ''
        if self.leftchild != None:
            leftstr = str(self.leftchild)
        if self.rightchild != None:
            rightstr = str(self.rightchild)
        if self.feature >= 0:
            return "{%s:%s|%s%s%s}" % (
                    self.variables[self.feature]
                    + (leftstr, str(self.data), rightstr))
        return "{<<leaf>>|%s}" % (str(self.data))

    def __repr__(self):

        return self.__str__()

    def set_left(self, child):

        if child == self:
            raise ValueError("Attempted to set self as left child!")
        self.leftchild = child
        if child != None:
            child.parent = self

    def set_right(self, child):

        if child == self:
            raise ValueError("Attempted to set self as right child!")
        self.rightchild = child
        if child != None:
            child.parent = self

    def is_leaf(self):

        return self.leftchild == None and self.rightchild == None

    def is_complete(self):

        return self.leftchild != None and self.rightchild != None

    def depth(self):

        leftdepth = 0
        if self.leftchild != None:
            leftdepth = self.leftchild.depth() + 1
        rightdepth = 0
        if self.rightchild != None:
            rightdepth = self.rightchild.depth() + 1
        return max(leftdepth, rightdepth)

    def balance(self):

        leftdepth = 0
        rightdepth = 0
        if self.leftchild != None:
            leftdepth = self.leftchild.depth() + 1
        if self.rightchild != None:
            rightdepth = self.rightchild.depth() + 1
        return rightdepth - leftdepth

    def get_leaves(self):

        if self.is_leaf():
            return [self]
        leftleaves = []
        if self.leftchild != None:
            leftleaves = self.leftchild.get_leaves()
        rightleaves = []
        if self.rightchild != None:
            rightleaves = self.rightchild.get_leaves()
        return leftleaves + rightleaves

    def get_incomplete_children(self):

        children = []
        if not self.is_complete():
            children.append(self)
        if self.leftchild != None:
            children += self.leftchild.get_incomplete_children()
        if self.rightchild != None:
            children += self.rightchild.get_incomplete_children()
        return children

    def walk(self, expression=None):

        if expression == None:
            expression = Cut()
        if self.feature < 0:
            if expression:
                yield expression
        if not self.forbidleft:
            leftcondition = Cut("%s<=%s" %
                    (self.variables[self.feature][0], self.data))
            newcondition = expression & leftcondition
            if self.leftchild != None:
                for condition in self.leftchild.walk(newcondition):
                    yield condition
            else:
                yield newcondition
        if not self.forbidright:
            rightcondition = Cut("%s>%s" %
                    (self.variables[self.feature][0], self.data))
            newcondition = expression & rightcondition
            if self.rightchild != None:
                for condition in self.rightchild.walk(newcondition):
                    yield condition
            else:
                yield newcondition
