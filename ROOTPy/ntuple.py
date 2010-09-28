import ROOT
from types import *
import re

class Ntuple(ROOT.TTree):

    def __init__(self, name, buffer=None, variables=None):

        ROOT.TTree.__init__(self,name,name)
        if buffer != None:
            if variables == None:
                variables = buffer.keys()
            for variable in variables:
                value = buffer[variable]
                if isinstance(value,Variable):
                    self.Branch(variable, value, "%s/%s"%(name,value.type()))
                else: # Must be a ROOT.vector
                    self.Branch(variable, value)

class NtupleChain:
    
    def __init__(self, treeName, files, buffer=None):
        
        self.treeName = treeName
        if type(files) is not list:
            files = [files]
        self.files = files
        self.buffer = buffer
        if self.buffer:
            for name,value in self.buffer.items():
                if name not in dir(self):
                    setattr(self,name,value)
                else:
                    raise ValueError("Illegal or duplicate branch name: %s"%name)
        self.weight = 1.
        self.tree = None
        self.file = None
        self.entry = 0
        self.entries = 0
        
    def _initialize(self):

        if self.tree != None:
            self.tree = None
        if self.file != None:
            self.file.Close()
            self.file = None
        if len(self.files) > 0:
            fileName = self.files.pop()
            self.file = ROOT.TFile.Open(fileName)
            if not self.file:
                print "WARNING: Skipping file. Could not open file %s"%(fileName)
                return self._initialize()
            self.tree = self.file.Get(self.treeName)
            if not self.tree:
                print "WARNING: Skipping file. Tree %s does not exist in file %s"%(self.treeName,fileName)
                return self._initialize()
            # Buggy D3PD:
            if len(self.tree.GetListOfBranches()) == 0:
                # Try the next file:
                print "WARNING: skipping tree with no branches in file %s"%fileName
                return self._initialize()
            self.entry = 0
            self.entries = self.tree.GetEntries()
            if self.buffer:
                self.tree.SetBranchStatus("*",False)
                for branch,address in self.buffer.items():
                    if not self.tree.GetBranch(branch):
                        print "WARNING: Skipping file. Branch %s was not found in tree %s in file %s"%(branch,self.treeName,fileName)
                        return self._initialize()
                    self.tree.SetBranchStatus(branch,True)
                    self.tree.SetBranchAddress(branch,address)
            return True
        return False

    def show(self):

        if self.tree:
            self.tree.Show()
    
    def read(self):
        
        if not self.entry < self.entries:
            if not self._initialize():
                return False
        self.tree.GetEntry(self.entry)
        self.weight = self.tree.GetWeight()
        self.entry += 1
        return True

class NtupleBuffer(dict):

    demote = {"Float_T":"F",
              "Int_T":"I",
              "Int":"I",
              "Float":"F",
              "F":"F",
              "I":"I",
              "UI":"UI",
              "vector<float>":"F",
              "vector<int>":"I",
              "vector<int,allocator<int> >":"I",
              "vector<float,allocator<float> >":"F",
              "VF":"F",
              "VI":"I",
              "vector<vector<float> >":"VF",
              "vector<vector<float> >":"VI",
              "vector<vector<int>,allocator<vector<int> > >":"VI",
              "vector<vector<float>,allocator<vector<float> > >":"VF",
              "VVF":"VF",
              "VVI":"VI"} 

    def __init__(self,variables,default=-1111,flatten=False):
        
        data = {}
        methods = dir(self)
        processed = []
        for name,type in variables:
            if flatten:
                type = NtupleBuffer.demote[type]
            if name in processed:
                raise ValueError("Duplicate variable name %s"%name)
            else:
                processed.append(name)
            if type.upper() in ("I","INT_T"):
                data[name] = Int(default)
            elif type.upper() in ("UI","UINT_T"):
                data[name] = UInt(default)
            elif type.upper() in ("F","FLOAT_T"):
                data[name] = Float(default)
            elif type.upper() in ("VI","VECTOR<INT>"):
                data[name] = ROOT.vector("int")()
            elif type.upper() in ("VF","VECTOR<FLOAT>"):
                data[name] = ROOT.vector("float")()
            elif type.upper() in ("VVI","VECTOR<VECTOR<INT> >"):
                data[name] = ROOT.vector("vector<int>")()
            elif type.upper() in ("VVF","VECTOR<VECTOR<FLOAT> >"):
                data[name] = ROOT.vector("vector<float>")()
            else:
                raise TypeError("Unsupported variable type: %s"%(type.upper()))
            if name not in methods and not name.startswith("_"):
                setattr(self,name,data[name])
            else:
                raise ValueError("Illegal variable name: %s"%name)
        dict.__init__(self,data)

    def reset(self):
        
        for value in self.values():
            value.clear()

    def __str__(self):

        return self.__repr__()

    def __repr__(self):

        rep = ""
        for var,value in self.items():
            rep += "%s ==> %s\n"%(var,value)
        return rep

# inTree is an existing tree containing data (entries>0).
# outTree is a new tree, not necessarily containing any branches, and should not contain any data (entries==0).
class NtupleProcessor(object):

    def __init__(self,inTree,outTree,inVars=None,outVars=None,flatten=False):

        self.inTree = inTree
        self.outTree = outTree
        self.inVars = inVars
        if not self.inVars:
            self.inVars = [(branch.GetName(),branch.GetListOfLeaves()[0].GetTypeName().upper()) for branch in inTree.GetListOfBranches()]
        self.outVars = outVars
        if not self.outVars:
            self.outVars = self.inVars
        self.inBuffer = NtupleBuffer(self.inVars)
        self.outBuffer = self.inBuffer
        self.inBuffer.fuse(self.inTree,createMissing=False)
        self.outBuffer.fuse(self.outTree,createMissing=True)
        self.entries = self.inTree.GetEntries()
        self.entry = 0
        self.flatten = flatten

    def read(self):

        if self.entry < self.entries:
            self.inTree.GetEntry(self.entry)
            return True
        return False

    def write(self):

        self.outTree.Fill()

    def copy(self):

        if self.flatten:
            while self.next():
                self.write()
        else:
            while self.next():
                self.write()

class NtupleReader:
    
    def __init__(self, treeList, branchMap, branchList=None, subs=None):
        
        if type(treeList) is not list:
            treeList = [treeList]
        assert(len(treeList)>0)
        self.treeList = [tree for tree in treeList]
        self.branchMap = branchMap
        self.subs = subs
        
        if not branchList:
            self.branchList = self.branchMap.keys()
        else:
            self.branchList = branchList
            
        self.weight = 1.
        self.tree = None
        self.entry = 0
        self.entries = 0
        
    def initialize(self):

        if self.tree != None:
            self.tree.ResetBranchAddresses()
        if len(self.treeList) > 0:
            self.tree = self.treeList.pop()
            self.entry = 0
            self.entries = self.tree.GetEntries()
            for branch in self.branchList:
                subBranch = branch
                if self.subs:
                    if branch in self.subs.keys():
                        subBranch = self.subs[branch]
                if not self.tree.GetBranch(subBranch):
                    raise RuntimeError("Branch %s was not found in tree %s"%(subBranch,self.tree.GetName()))
                self.tree.SetBranchAddress(subBranch,self.branchMap[branch].address())
            return True
        return False
    
    def isReady(self):
        
        return self.entry < self.entries
    
    def read(self):
        
        if not self.isReady():
            if not self.initialize():
                return False
        self.tree.GetEntry(self.entry)
        self.weight = self.tree.GetWeight()
        self.entry += 1
        return True

class FastTuple:

    import numpy as np
    
    def __init__(self,trees,branchNames=None):
        
        if branchNames != None:
            if type(branchNames) is not list:
                branchNames = [branchNames]
        self.specialBranchNames = ["__weight"]
        self.branchNames = branchNames
        
        if self.branchNames == None: 
            self.branchNames = [branch.GetName() for branch in trees[0].GetListOfBranches()]
        branches = dict([(name,[]) for name in self.branchNames + self.specialBranchNames])
        buffer = dict([(name,Float()) for name in self.branchNames])
        
        #read in trees as lists
        reader = NtupleReader(trees,buffer)
        while reader.read():
            for name in self.branchNames:
                branches["__weight"].append(reader.weight)
                branches[name].append(buffer[name].value())
        
        #convert to numpy array
        self.arrays = dict([(name,np.array(branches[name])) for name in self.branchNames])
    
    def sort(self,branch):

        if self.arrays.has_key(branch):
            inx = np.argsort(self.arrays[branch])
            for key in self.arrays.keys():
                self.arrays[key] = np.array([self.arrays[key][i] for i in inx])
    
    def getListOfBranches(self):
        
        return self.arrays.keys()
    
    def getBranch(self,name):
        
        if self.arrays.has_key(name):
            return self.arrays[name]
        return None
    
    """
    def apply_cut(self,name,low=None,high=None):
        
        if name not in self.branchToIndex.keys():
            return
        index = self.branchToIndex[name]
        if low != None and high != None:
            condition = (self.crop[index] >= low) & (self.crop[index] < high)
        elif low != None:
            condition = self.crop[index] >= low
        elif high != None:
            condition = self.crop[index] < high
        else:
            return
        self.crop = self.crop.compress(condition,axis=1)
    
    def apply_cuts(self,cuts):
        
        self.reset()
        for cut in cuts:
            self.apply_cut(cut["variable"],low=cut["low"],high=cut["high"])
    """

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


class Cut:
    
    operator_dict = {
        "==":{"negate":"!=","flip":"=="},
        "!=":{"negate":"==","flip":"!="},
        "<":{"negate":">=","flip":">"},
        ">":{"negate":"<=","flip":"<"},
        "<=":{"negate":">","flip":">="},
        ">=":{"negate":"<","flip":"<="}
    }
           
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
            node.content = Cut.operator_dict[node.content]["negate"]
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
                node.content = Cut.operator_dict[node.content]["flip"]
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
            return self.clone()
        else:
            newCut = self.clone()
            self.recursive_replace(newCut.root,oldVariable,newVariable)
            return newCut

    def recursive_replace(self,node,oldVariable,newVariable):

        if not node:
            return
        if node.type == self.operand:
            if node.content == oldVariable:
                node.content = newVariable
        else:
            self.recursive_replace(node.left,oldVariable,newVariable)
            self.recursive_replace(node.right,oldVariable,newVariable)
    
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
