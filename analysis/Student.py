import uuid
import os
import ROOT
from FilterList import *

class Student(object):

    def __init__(self):

        self.name = uuid.uuid4().hex
        self.output = ROOT.TFile.Open("%s.root"%self.name,"recreate")
        self.filters = FilterList()
        
    def coursework(self): pass

    def research(self): pass

    def defend(self):

        self.output.Write()
        self.output.Close()
