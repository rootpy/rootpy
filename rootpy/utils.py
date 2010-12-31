import ROOT
from rootpy.core import Object, Plottable
from registry import lookup

def asrootpy(tobject):

    # is this object already converted?
    if isinstance(tobject, Object):
        return tobject
    template = Plottable()
    template.decorate(tobject)
    
    cls, inits = lookup(tobject.__class__)
    tobject.__class__ = cls
    for init in inits:
        init(tobject)

    """
    if isinstance(tobject, ROOT.TTree):
        tobject.__class__ = Ntuple
    elif isinstance(tobject, ROOT.TH1):
        tobject.__class__ = plotting._Hist_class(rootclass = tobject.__class__)
        tobject._post_init()
    elif isinstance(tobject, ROOT.TH2):
        tobject.__class__ = plotting._Hist2D_class(rootclass = tobject.__class__)
        tobject._post_init()
    elif isinstance(tobject, ROOT.TH3):
        tobject.__class__ = plotting._Hist3D_class(rootclass = tobject.__class__)
        tobject._post_init()
    elif isinstance(tobject, ROOT.TGraphAsymmErrors):
        tobject.__class__ = plotting.Graph
    """
    
    
    if isinstance(tobject, Plottable):
        tobject.decorate(template)
    return tobject
