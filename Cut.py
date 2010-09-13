import re

class Node:
    
    def __init__(self):
        
        self.type = None
        self.content = None
        self.parent = None
        self.left = None
        self.right = None
        self.isRightChild = False
        
    def clone(self,parent=None):
        
        node = Node()
        node.type = self.type
        node.content = self.content
        node.parent = parent
        node.isRightChild = self.isRightChild
        if self.left != None:
            node.left = self.left.clone(node)
        if self.right != None:
            node.right = self.right.clone(node)
        return node

    def setLeftChild(self,node):

        self.left = node
        node.isRightChild = False
        node.parent = self

    def setRightChild(self,node):

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

operator_dict = {
    "==":{"negate":"!=","flip":"=="},
    "!=":{"negate":"==","flip":"!="},
    "<":{"negate":">=","flip":">"},
    ">":{"negate":"<=","flip":"<"},
    "<=":{"negate":">","flip":">="},
    ">=":{"negate":"<","flip":"<="}
}

class Cut:
        
    def __init__(self,cut="",debug=False):
        
        self.numericOperand = re.compile('^[\+\-]?[0-9.]+$')
        self.namedOperand = re.compile('[a-zA-Z\_]+')
        self.debug = debug
        self.operand = re.compile('^([\+\-]?[a-zA-Z0-9.\%\_]+)')
        self.operator = re.compile('^(\!=|<=|>=|==|>|<)')
        self.logical = re.compile('^(\&\&|\|\|)')
        self.precedence = [self.logical,self.operator]
        self.actsOn = {self.logical:[self.logical,self.operator],
                       self.operator:[self.operand],
                       self.operand:[],
                       "open":[]}
        
        if not cut:
            cut=""
        if isinstance(cut,Node):
            self.root = cut
        elif type(cut) in [str,unicode]:
            if cut == "":
                self.root = None
            else:
                self.root = self.makeTree(cut,debug)
                if not self.root:
                    raise Warning("expression %s is not well-formed"%cut)
        else:
            print "%s %s"%(type(cut),cut)
            raise TypeError("cut parameter must be of type str or Node")
    
    def clone(self):
        
        if self.root != None:
            return Cut(self.root.clone())
        return Cut()
    
    def __and__(self, other):
        
        if type(other) in [str,unicode]:
            other = Cut(other)
        
        return self.join(self,other,"&&")

    def __mul__(self, other):

        if type(other) in [str,unicode]:
            other = Cut(other)
        
        return self.__and__(other)
    
    def __or__(self, other):

        if type(other) in [str,unicode]:
            other = Cut(other)
        
        return self.join(self,other,"||")
    
    def __add__(self, other):

        if type(other) in [str,unicode]:
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

    def recursive_negate(self,node):

        if node == None:
            return
        if node.type == self.logical:
            if node.content == "||":
                node.content = "&&"
            elif node.content == "&&":
                node.content = "||"
        if node.type == self.operator:
            node.content = operator_dict[node.content]["negate"]
        self.recursive_negate(node.left)
        self.recursive_negate(node.right)
    
    def order(self,node=None):
        
        if node == None:
            node = self.root
        if node.type == self.operator:
            left = node.left
            right = node.right
            if re.match(self.numericOperand,left.content):
                temp = left.content
                left.content = right.content
                right.content = temp
                node.content = operator_dict[node.content]["flip"]
            return
        self.order(node.left)
        self.order(node.right)

    def join(self,left,right,logic):
        
        assert(logic in ["&&","||"])
        if not left.root:
            if right.root:
                return Cut(right.root.clone())
            return Cut()
        elif not right.root:
            if left.root:
                return Cut(left.root.clone())
            return Cut()
        else:
            node = Node()
            node.type = self.logical
            node.content = logic
            leftCopy = left.root.clone()
            rightCopy = right.root.clone()
            node.setLeftChild(leftCopy)
            node.setRightChild(rightCopy)
            return Cut(node)
    
    def __str__(self):
        
        return self.__repr__()
    
    def __repr__(self):
        
        if not self.root:
            return "Empty Cut"
        else:
            return self.infix(self.root)
    
    def empty(self):
        
        return not self.root
        
    def safeString(self):
        
        if self.empty():
            return ""
        string = self.__str__()
        string = string.replace("==","-eq-")
        string = string.replace("<=","-leq-")
        string = string.replace(">=","-geq-")
        string = string.replace("<","-lt-")
        string = string.replace(">","-gt-")
        string = string.replace("&&","-and-")
        string = string.replace("||","-or-")
        string = string.replace("(","L")
        string = string.replace(")","R")
        return string

    def LaTeX(self):
        
        if self.empty():
            return ""
        string = self.__str__()
        string = string.replace("==","=")
        string = string.replace("<=","\leq")
        string = string.replace(">=","\geq")
        string = string.replace("&&"," and ")
        string = string.replace("||"," or ")
        return string
    
    def getNumeric(self):
        
        if self.root.type == self.operator:
            if re.match(numericOperand,self.root.left.content):
                return float(self.root.left.content)
            if re.match(numericOperand,self.root.right.content):
                return float(self.root.right.content)
        else:
            return None
    
    def substitute(self,oldVariable,newVariable):
        
        if self.empty():
            return self
        else:
            return Cut(self.__repr__().replace(oldVariable,newVariable))
    
    def infix(self,node):
        
        if node.type == self.logical:
            return "(%s)%s(%s)"%(self.infix(node.left),node.content,self.infix(node.right))
        elif node.type == self.operator:
            return "%s%s%s"%(node.left.content,node.content,node.right.content)
        else:
            return "ERROR"
    
    def removeAll(self,name,currNode=None):
        cuts = []
        if not currNode:
            currNode = self.root
        if not currNode:
            return cuts
        if currNode.type == self.operator:
            if currNode.left.content == name or currNode.right.content == name:
                if currNode == self.root:
                    self.root = None
                newroot = currNode.remove()
                if newroot != None:
                    self.root = newroot
                cuts.append(Cut(currNode))
            return cuts
        cuts += self.removeAll(name,currNode.left)
        cuts += self.removeAll(name,currNode.right)
        return cuts
    
    def makeTree(self,expression,debug=False):
        
        stack = []
        while len(expression) > 0:
            if len(stack)>=3:
                ok = True
                for node in stack[-3:]:
                    if node.type not in [self.logical,self.operator]:
                        ok = False
                if ok:
                    right = stack.pop()
                    root = stack.pop()
                    left = stack.pop()
                    root.setLeftChild(left)
                    root.setRightChild(right)
                    stack.append(root)
            if expression[0]==' ':
                expression = expression[1:]
                continue
            if debug:
                print stack
            if expression[0]=='(':
                if len(stack) > 0:
                    if stack[-1].type not in [self.precedence[0],"open"]:
                        return None
                node = Node()
                node.type="open"
                node.content = "("
                stack.append(node)
                expression = expression[1:]
                continue
            if expression[0]==')':
                if len(stack) in [0,1]:
                    return None
                if stack[-2].type != "open":
                    return None
                stack.pop(-2)
                if len(stack) >= 3:
                    if stack[-2].type == self.precedence[0]:
                        right = stack.pop()
                        root = stack.pop()
                        left = stack.pop()
                        root.setLeftChild(left)
                        root.setRightChild(right)
                        stack.append(root)
                expression = expression[1:]
                continue
            operandMatch = re.match(self.operand,expression)
            if operandMatch:
                if debug:
                    print "operand: %s stack: %s"%(operandMatch.group(),stack)
                node = Node()
                node.type = self.operand
                node.content = operandMatch.group(0)
                if len(stack) == 0:
                    stack.append(node)
                elif stack[-1].type in ["open",self.logical]:
                    stack.append(node)
                elif self.operand in self.actsOn[stack[-1].type]:
                    op = stack.pop()
                    if len(stack) == 0:
                        return None
                    left = stack.pop()
                    if re.match(self.numericOperand,node.content) and re.match(self.numericOperand,left.content):
                        return None
                    op.setLeftChild(left)
                    op.setRightChild(node)
                    stack.append(op)
                else:
                    return None
                expression = expression[len(operandMatch.group()):]
                continue
            found = False
            for operator in self.precedence:    
                match = re.match(operator,expression)
                if match:
                    if debug:
                        print "operator: %s stack: %s"%(match.group(),stack)
                    node = Node()
                    node.type = operator
                    node.content = match.group(0)
                    if len(stack) == 0:
                        return None
                    elif stack[-1].type in self.actsOn[operator]:
                        stack.append(node)
                    else:
                        return None
                    expression = expression[len(match.group()):]
                    found = True
                    break
            if not found:
                return None
        if len(stack)>=3:
            ok = True
            for node in stack[-3:]:
                if node.type not in [self.logical,self.operator]:
                    ok = False
            if ok:
                right = stack.pop()
                root = stack.pop()
                left = stack.pop()
                root.setLeftChild(left)
                root.setRightChild(right)
                stack.append(root)
        if len(stack) > 1:
            if debug:
                print stack
                print "Stack has more than one element and expression is fully parsed!"
            return None
        if debug:
            print stack
        return stack.pop()
