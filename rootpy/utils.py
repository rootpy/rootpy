import ROOT
import plotting

def asrootpy(tobject):

    if isinstance(tobject, ROOT.TH1):
        template = _Plottable()
        template.decorate(tobject)
        tobject.__class__ = plotting._Hist_class(rootclass = tobject.__class__)
        tobject._post_init()
        tobject.decorate(template)
    elif isinstance(tobject, ROOT.TH2):
        template = _Plottable()
        template.decorate(tobject)
        tobject.__class__ = plotting._Hist2D_class(rootclass = tobject.__class__)
        tobject._post_init()
        tobject.decorate(template)
    elif isinstance(tobject, ROOT.TH3):
        template = _Plottable()
        template.decorate(tobject)
        tobject.__class__ = plotting._Hist3D_class(rootclass = tobject.__class__)
        tobject._post_init()
        tobject.decorate(template)
    elif isinstance(tobject, ROOT.TGraphAsymmErrors):
        template = _Plottable()
        template.decorate(tobject)
        tobject.__class__ = plotting.Graph
        tobject.decorate(template)
    return tobject
