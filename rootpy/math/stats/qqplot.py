# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
"""
Taken from exmaple by Zhiyi Liu, zhiyil@fnal.gov
here: http://root.cern.ch/phpBB3/viewtopic.php?f=3&t=6865
and converted into Python
"""

import ROOT
from math import sqrt
from array import array
from rootpy.plotting import Graph, Hist, Canvas


def qqplot(h1, h2, quantiles=None):
    """
    Return a Graph of a QQ plot and confidence band
    """
    if quantiles is None:
        quantiles = max(min(len(h1), len(h2)) / 2, 1)
    nq = quantiles
    xq = array('d', [0.] * nq)   # position where to compute the quantiles in [0,1]
    yq1 = array('d', [0.] * nq)  # array to contain the quantiles
    yq2 = array('d', [0.] * nq)  # array to contain the quantiles

    for i in xrange(nq):
        xq[i] = float(i + 1) / nq

    h1.GetQuantiles(nq, yq1, xq)
    h2.GetQuantiles(nq, yq2, xq)

    xq_plus = array('d', [0.] * nq)
    xq_minus = array('d', [0.] * nq)
    yq2_plus = array('d', [0.] * nq)
    yq2_minus = array('d', [0.] * nq)

    """
    KS_cv: KS critical value

               1.36
    KS_cv = -----------
             sqrt( N )

    Where 1.36 is for alpha = 0.05 (confidence level 1-5%=95%, about 2 sigma)

    For 1 sigma (alpha=0.32, CL=68%), the value in the nominator is 0.9561,
    it is gotten by GetCriticalValue(1, 1 - 0.68).

    NOTE:
    * For 1-sample KS test (data and theoretic), N should be n
    * For 2-sample KS test (2 data set), N should be sqrt(m*n/(m+n))! Here is the case
      m or n (size of samples) should be effective size for a histogram
    * Critical value here is valid for only for sample size >= 80 (some references say 35)
      which means, for example, for a unweighted histogram, it must have more than 80 (or 35)
      entries filled and then confidence band is reliable.
    """

    esum1 = effective_sample_size(h1)
    esum2 = effective_sample_size(h2)

    # one sigma band
    KS_cv = (critical_value(1, 1 - 0.68) /
             sqrt((esum1 * esum2) / (esum1 + esum2)))

    for i in xrange(nq):
        xq_plus[i] = float(xq[i] + KS_cv)   # upper limit
        xq_minus[i] = float(xq[i] - KS_cv)  # lower limit

    h2.GetQuantiles(nq, yq2_plus, xq_plus)
    h2.GetQuantiles(nq, yq2_minus, xq_minus)

    yq2_err_plus = array('d', [0.] * nq)
    yq2_err_minus = array('d', [0.] * nq)
    for i in xrange(nq):
        yq2_err_plus[i] = yq2_plus[i] - yq2[i]
        yq2_err_minus[i] = yq2[i] - yq2_minus[i]

    #forget the last point, so number of points: (nq-1)
    gr = Graph(nq - 1)
    for i in xrange(nq - 1):
        gr[i] = (yq1[i], yq2[i])
        # confidence level band
        gr.SetPointEYlow(i, yq2_err_minus[i])
        gr.SetPointEYhigh(i, yq2_err_plus[i])

    return gr


def effective_sample_size(h):
    """
    calculate effective sample size for a histogram
    same way as ROOT does.
    """
    axis = h.GetXaxis()
    last = axis.GetNbins()
    sum = 0
    ew = 0
    w = 0
    for bin in xrange(1, last + 1):
        sum += h.GetBinContent(bin)
        ew = h.GetBinError(bin)
        w += ew * ew
    esum = sum * sum / w
    return esum


def critical_value(n, p):
    """
    the routine is used to calculate critical value given
    n and p, confidential level = 1 - p.
    Original Reference:
    http://velveeta.che.wisc.edu/octave/lists/archive//octave-sources.2003/msg00031.html
    I just checked it, but it is not available now...
    """
    dn = 1
    delta = 0.5
    res = ROOT.TMath.KolmogorovProb(dn * sqrt(n))
    while res > 1.0001 * p or res < 0.9999 * p:
        if (res > 1.0001 * p):
            dn = dn + delta
        if (res < 0.9999 * p):
            dn = dn - delta
        delta = delta / 2.
        res = ROOT.TMath.KolmogorovProb(dn * sqrt(n))
    return dn


if __name__ == '__main__':
    """
    this is an example of drawing a quantile-quantile plot with
    confidential level (CL) band by Zhiyi Liu, zhiyil@fnal.gov
    """
    ROOT.gROOT.SetStyle("Plain")
    ROOT.gStyle.SetOptStat(0)
    can = Canvas(name="can", title="can", width=600, height=450)
    rand = ROOT.TRandom3()
    h1 = Hist(100, -5, 5, name="h1", title="Histogram 1")
    h1.Sumw2()
    h1.SetLineColor(ROOT.kRed)
    h2 = Hist(100, -5, 5, name="h2", title="Histogram 2")
    h2.SetLineColor(ROOT.kBlue)

    for ievt in xrange(10000):
        #some test histograms:
        #1. let 2 histograms screwed
        #h1.Fill(rand.Gaus(0.5, 0.8))
        #h2.Fill(rand.Gaus(0, 1))

        #2. long tail and short tail
        h1.Fill(rand.Gaus(0, 0.8))
        h2.Fill(rand.Gaus(0, 1))

    #hs = ROOT.THStack("hs", "2 example distributions")
    #hs.Add(h1)
    #hs.Add(h2)
    #hs.Draw("nostack")

    h1.Draw()
    h2.Draw('same')

    #draw legend
    leg = ROOT.TLegend(0.7, 0.7, 0.89, 0.89)
    leg.SetFillColor(0)
    leg.AddEntry(h1, h1.GetTitle(), "pl")
    leg.AddEntry(h2, h2.GetTitle(), "pl")
    leg.Draw()
    can.Modified()
    can.Update()

    gr = qqplot(h1, h2)

    gr.SetLineColor(ROOT.kRed + 2)
    gr.SetMarkerColor(ROOT.kRed + 2)
    gr.SetMarkerStyle(20)
    gr.SetTitle("QQ with CL")
    gr.GetXaxis().SetTitle(h1.GetTitle())
    gr.GetYaxis().SetTitle(h2.GetTitle())
    gr.SetFillColor(17)
    gr.SetFillStyle(1001)

    c = Canvas(name="c", title="QQ with CL", width=600, height=450)
    gr.Draw("ap")
    x_min = gr.GetXaxis().GetXmin()
    x_max = gr.GetXaxis().GetXmax()
    y_min = gr.GetXaxis().GetXmin()
    y_max = gr.GetXaxis().GetXmax()
    c.Clear()
    gr.Draw('a3')
    gr.Draw('Xp same')

    # a straight line y=x to be a reference
    f_dia = ROOT.TF1("f_dia", "x",
                     h1.GetXaxis().GetXmin(),
                     h1.GetXaxis().GetXmax())
    f_dia.SetLineColor(9)
    f_dia.SetLineWidth(2)
    f_dia.SetLineStyle(2)
    f_dia.Draw("same")

    leg = ROOT.TLegend(0.52, 0.15, 0.87, 0.35)
    leg.SetFillColor(0)
    leg.SetShadowColor(17)
    leg.SetBorderSize(3)
    leg.AddEntry(gr, "QQ points", "p")
    leg.AddEntry(gr, "68% CL band", "f")
    leg.AddEntry(f_dia, "Diagonal line", "l")
    leg.Draw()

    c.Modified()
    c.Update()
    c.OwnMembers()
    c.Draw()
    raw_input()
