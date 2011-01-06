import os
import re

class _CutNode:
    
    def __init__(self):
        
        self.type = None
        self.content = None
        self.parent = None
        self.left = None
        self.right = None
        self.isRightChild = False
        
    def clone(self, parent=None):
        
        node = _CutNode()
        node.type = self.type
        node.content = self.content
        node.parent = parent
        node.isRightChild = self.isRightChild
        if self.left != None:
            node.left = self.left.clone(node)
        if self.right != None:
            node.right = self.right.clone(node)
        return node

    def setLeftChild(self, node):

        self.left = node
        node.isRightChild = False
        node.parent = self

    def setRightChild(self, node):

        self.right = node
        node.isRightChild = True
        node.parent = self

    def getSibling(self):

        if self.parent != None:
            if self.isRightChild:
                return self.parent.getLeftChild()
            else:
                return self.parent.getRightChild()
        return None

    def remove(self):

        newroot = None
        if self.parent == None:
            return newroot
        elif self.parent.parent == None:
            if self.isRightChild:
                newroot = self.parent.left
            else:
                newroot = self.parent.right
        else:
            if self.isRightChild:
                if self.parent == self.parent.parent.left:
                    self.parent.parent.left = self.parent.left
                else:
                    self.parent.parent.right = self.parent.left
                self.parent.left.parent = self.parent.parent
            else:
                if self.parent == self.parent.parent.left:
                    self.parent.parent.left = self.parent.right
                else:
                    self.parent.parent.right = self.parent.right
                self.parent.right.parent = self.parent.parent
        self.parent = None
        return newroot
        
    def __str__(self):
        
        return self.__repr__()
    
    def __repr__(self):
        
        return self.content

"""
negate
open
close
operand (named, numeric, function)
operator (arithmetic, compare, logical)
"""

class Cut:
    
    operator_dict = {
        "==": {"negate": "!=", "flip": "=="},
        "!=": {"negate": "==", "flip": "!="},
        "<":  {"negate": ">=", "flip": ">"},
        ">":  {"negate": "<=", "flip": "<"},
        "<=": {"negate": ">",  "flip": ">="},
        ">=": {"negate": "<",  "flip": "<="}
    }
    
    numeric_operand = re.compile(
    """(?x)
       ^
          [+-]?\ *      # first, match an optional sign *and space*
          (             # then match integers or f.p. mantissas:
              \d+       # start out with a ...
              (
                  \.\d* # mantissa of the form a.b or a.
              )?        # ? takes care of integers of the form a
             |\.\d+     # mantissa of the form .b
          )
          ([eE][+-]?\d+)?  # finally, optionally match an exponent
    """)
    named_operand = re.compile('^[a-zA-Z][a-zA-Z0-9()_]*(\%\d+)?')
    operator = re.compile('^(\!=|<=|>=|==|>|<|\+|\-|/|\*)')
    logical = re.compile('^(\&\&|\|\|)')
    precedence = [logical, operator]
    acts_on = {logical:         [logical, operator, named_operand],
               operator:        [operator, named_operand, numeric_operand],
               named_operand:   [],
               numeric_operand: [],
               "open":          [],
               "negate":        [logical, operator, named_operand]}
  
    def __init__(self, cut = "", debug = False):
        
        self.debug = debug        
        if not cut:
            cut=""
        if isinstance(cut, _CutNode):
            self.root = cut
        elif type(cut) in [str, unicode]:
            if cut == "":
                self.root = None
                return
            if os.path.isfile(cut):
                filename = cut
                print "Reading cut from file %s"%filename
                try:
                    file = open(filename, 'r')
                    cut = "".join(line.strip() for line in file.readlines())
                    file.close()
                except:
                    print "unable to read cut from file %s"%filename
                    self.root = None
                    return
            self.root = self.build(cut, debug)
            if not self.root:
                raise Warning("expression %s is not well-formed"%cut)
        else:
            print "%s %s"%(type(cut), cut)
            raise TypeError("cut parameter must be of type str or _CutNode")
    
    def clone(self):
        
        if self.root != None:
            return Cut(self.root.clone())
        return Cut()
    
    def __and__(self, other):
        
        if type(other) in [str, unicode]:
            other = Cut(other)
        
        return self.join(self, other, "&&")

    def __mul__(self, other):

        if type(other) in [str, unicode]:
            other = Cut(other)
        
        return self.__and__(other)
    
    def __or__(self, other):

        if type(other) in [str, unicode]:
            other = Cut(other)
        
        return self.join(self, other, "||")
    
    def __add__(self, other):

        if type(other) in [str, unicode]:
            other = Cut(other)
        
        return self.__or__(other)

    def __neg__(self):

        if self.root != None:
            newroot = self.root.clone()
            self.recursive_negate(newroot)
            return Cut(newroot)
        return Cut()

    def __pos__(self):

        if self.root != None:
            return Cut(self.root.clone())
        return Cut("")

    def recursive_negate(self, node):

        if node == None:
            return
        if node.type == Cut.logical:
            if node.content == "||":
                node.content = "&&"
            elif node.content == "&&":
                node.content = "||"
        if node.type == Cut.operator:
            node.content = Cut.operator_dict[node.content]["negate"]
        self.recursive_negate(node.left)
        self.recursive_negate(node.right)
    
    def order(self, node=None):
        
        if node == None:
            node = self.root
        if node.type == Cut.operator:
            left = node.left
            right = node.right
            if re.match(Cut.numericOperand, left.content):
                temp = left.content
                left.content = right.content
                right.content = temp
                node.content = Cut.operator_dict[node.content]["flip"]
            return
        self.order(node.left)
        self.order(node.right)

    def join(self, left, right, logic):
        
        assert(logic in ["&&", "||"])
        if not left.root:
            if right.root:
                return Cut(right.root.clone())
            return Cut()
        elif not right.root:
            if left.root:
                return Cut(left.root.clone())
            return Cut()
        else:
            node = _CutNode()
            node.type = Cut.logical
            node.content = logic
            leftCopy = left.root.clone()
            rightCopy = right.root.clone()
            node.setLeftChild(leftCopy)
            node.setRightChild(rightCopy)
            return Cut(node)
    
    def __str__(self):
        
        return self.__repr__()
    
    def __repr__(self):
        
        if self.root == None:
            return ""
        else:
            return self.infix(self.root)
    
    def __nonzero__(self):

        return not self.empty()
    
    def empty(self):
        
        return self.root == None
        
    def safeString(self):
        
        if self.empty():
            return ""
        string = self.__str__()
        string = string.replace("==", "-eq-")
        string = string.replace("<=", "-leq-")
        string = string.replace(">=", "-geq-")
        string = string.replace("<", "-lt-")
        string = string.replace(">", "-gt-")
        string = string.replace("&&", "-and-")
        string = string.replace("||", "-or-")
        string = string.replace("(", "L")
        string = string.replace(")", "R")
        return string

    def LaTeX(self):
        
        if self.empty():
            return ""
        string = self.__str__()
        string = string.replace("==", "=")
        string = string.replace("<=", "\leq")
        string = string.replace(">=", "\geq")
        string = string.replace("&&", " and ")
        string = string.replace("||", " or ")
        return string
    
    def getNumeric(self):
        
        if self.root.type == Cut.operator:
            if re.match(numericOperand, self.root.left.content):
                return float(self.root.left.content)
            if re.match(numericOperand, self.root.right.content):
                return float(self.root.right.content)
        else:
            return None
    
    def substitute(self, oldVariable, newVariable):
        
        if self.empty():
            return self.clone()
        else:
            newCut = self.clone()
            self.recursive_replace(newCut.root, oldVariable, newVariable)
            return newCut

    def recursive_replace(self, node, oldVariable, newVariable):

        if not node:
            return
        if node.type == Cut.operand:
            if node.content == oldVariable:
                node.content = newVariable
        else:
            self.recursive_replace(node.left, oldVariable, newVariable)
            self.recursive_replace(node.right, oldVariable, newVariable)
    
    def recurse(self, node, function, *args):

        if not node:
            return
        if not function(node, *args):
            self.recursive_operation(node.left, function, *args)
            self.recursive_operation(node.right, function, *args)

    def set_cut_on(self, variable, value):
        
        def f(node, var, val):
            if node.content == Cut.operand:
                return False
        self.recursive_operation(self.root, f, variable, value)
        
    
    def infix(self, node):
        
        if node.type == Cut.logical:
            return "(%s)%s(%s)"%(self.infix(node.left), node.content, self.infix(node.right))
        elif node.type == Cut.operator:
            return "%s%s%s"%(node.left.content, node.content, node.right.content)
        else:
            return node.content
    
    def removeAll(self, name, curr_CutNode=None):
        cuts = []
        if not curr_CutNode:
            curr_CutNode = self.root
        if not curr_CutNode:
            return cuts
        if curr_CutNode.type == Cut.operator:
            if curr_CutNode.left.content == name or curr_CutNode.right.content == name:
                if curr_CutNode == self.root:
                    self.root = None
                newroot = curr_CutNode.remove()
                if newroot != None:
                    self.root = newroot
                cuts.append(Cut(curr_CutNode))
            return cuts
        cuts += self.removeAll(name, curr_CutNode.left)
        cuts += self.removeAll(name, curr_CutNode.right)
        return cuts
    
    def build(self, expression, debug=False):
        
        stack = []
        while len(stack) != 1 or len(expression) > 0:
            while len(stack)>=2:
                if stack[-2].type == "negate" and stack[-1].type in [Cut.logical, Cut.operator]:
                    self.recursive_negate(stack[-1])
                    stack.pop(-2)
                    continue
                break
            if len(stack)>=3:
                if stack[-2].type in [Cut.logical, Cut.operator]:
                    right = stack.pop()
                    root = stack.pop()
                    left = stack.pop()
                    root.setLeftChild(left)
                    root.setRightChild(right)
                    stack.append(root)
            if len(expression) == 0 and len(stack) == 1:
                break
            elif len(expression) == 0:
                return None
            if expression[0]==' ':
                expression = expression[1:]
                continue
            if debug:
                print stack
            if expression[0]=='(':
                if len(stack) > 0:
                    if stack[-1].type not in [Cut.precedence[0], "open", "negate"]:
                        return None
                node = _CutNode()
                node.type="open"
                node.content = "("
                stack.append(node)
                expression = expression[1:]
                continue
            if expression[0]==')':
                if len(stack) in [0, 1]:
                    return None
                if stack[-2].type != "open":
                    return None
                stack.pop(-2)
                if len(stack) >= 3:
                    if stack[-2].type == Cut.precedence[0]:
                        right = stack.pop()
                        root = stack.pop()
                        left = stack.pop()
                        root.setLeftChild(left)
                        root.setRightChild(right)
                        stack.append(root)
                expression = expression[1:]
                continue
            namedOperandMatch = re.match(Cut.named_operand, expression)
            numericOperandMatch = re.match(Cut.numeric_operand, expression)
            if namedOperandMatch or numericOperandMatch:
                if debug:
                    if namedOperandMatch:
                        print "operand: %s stack: %s"%(namedOperandMatch.group(), stack)
                    else:
                        print "operand: %s stack: %s"%(numericOperandMatch.group(), stack)
                node = _CutNode()
                if namedOperandMatch:
                    node.type = Cut.named_operand
                    node.content = namedOperandMatch.group(0)
                else:
                    node.type = Cut.numeric_operand
                    node.content = numericOperandMatch.group(0)
                if len(stack) == 0:
                    stack.append(node)
                elif stack[-1].type in ["open", Cut.logical]:
                    stack.append(node)
                elif Cut.named_operand in Cut.acts_on[stack[-1].type] or Cut.numeric_operand in Cut.acts_on[stack[-1].type]:
                    op = stack.pop()
                    if len(stack) == 0:
                        return None
                    left = stack.pop()
                    if node.type == Cut.numeric_operand and left.type == Cut.numeric_operand:
                        return None
                    op.setLeftChild(left)
                    op.setRightChild(node)
                    stack.append(op)
                else:
                    return None
                if namedOperandMatch:
                    expression = expression[len(namedOperandMatch.group()):]
                else:
                    expression = expression[len(numericOperandMatch.group()):]
                continue
            found = False
            for operator in Cut.precedence:    
                match = re.match(operator, expression)
                if match:
                    if debug:
                        print "operator: %s stack: %s"%(match.group(), stack)
                    node = _CutNode()
                    node.type = operator
                    node.content = match.group(0)
                    if len(stack) == 0:
                        return None
                    elif stack[-1].type in Cut.acts_on[operator]:
                        stack.append(node)
                    else:
                        return None
                    expression = expression[len(match.group()):]
                    found = True
                    break
            if expression[0] == '!': # negation
                node = _CutNode()
                node.type = "negate"
                node.content = '!'
                stack.append(node)
                expression = expression[1:]
                found = True
            if not found:
                return None
        if debug:
            print stack
        return stack.pop()
