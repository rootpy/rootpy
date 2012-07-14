import ROOT

f = ROOT.TFile.Open('ttree.root')
t = f.Get('ParTree_Postselect')
#t.GetEntry(0)

for branch in t.GetListOfBranches():
    print branch
    print branch.GetClassName()
    print branch.GetNleaves()
    print branch.GetListOfLeaves()[0].GetTypeName()
    for leave in branch.GetListOfLeaves():
        print leave.GetNdata()
        N = leave.GetLen()
        print N
        print "#####"
        for i in range(N):
            print leave.GetValue(i)
        print "--------"
    print "==============="
