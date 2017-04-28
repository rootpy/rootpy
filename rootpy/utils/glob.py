"""
Reproduce the standard glob package behaviour but use TSystem to be able to
query remote file systems, like xrootd
"""
from __future__ import print_function
from rootpy.ROOT import gSystem
import glob as gl
import os.path
import fnmatch


def __directory_iter(directory):
    while True:
        file = gSystem.GetDirEntry(directory)
        if not file:
            break
        yield file
    return


def glob(pathname):
    # Let normal python glob try first
    try_glob = gl.glob(pathname)
    if try_glob:
        return try_glob

    # If pathname does not contain a wildcard:
    if not gl.has_magic(pathname):
        return [pathname]

    # Split the pathname into a directory and basename
    # (which should include the wild-card)
    dirname, basename = os.path.split(pathname)

    # Uses `TSystem` to open the directory.
    # TSystem itself wraps up the calls needed to query xrootd.
    dirname = gSystem.ExpandPathName(dirname)
    directory = gSystem.OpenDirectory(dirname)

    files = []
    if directory:
        for file in __directory_iter(directory):
            if file in [".", ".."]:
                continue
            if not fnmatch.fnmatchcase(file, basename):
                continue
            files.append(os.path.join(dirname, file))
        gSystem.FreeDirectory(directory)
    return files


def iglob(pathname):
    for name in glob(pathname):
        yield name


if __name__ == "__main__":
    test_paths = [
        "data/*root",
        "data/L1Ntuple_test_3.root",
        """root://eoscms.cern.ch//eos/cms/store/group/dpg_trigger/"""
        """comm_trigger/L1Trigger/L1Menu2016/Stage2/"""
        """l1t-integration-v88p1-CMSSW-8021/SingleMuon/"""
        """crab_l1t-integration-v88p1-CMSSW-8021__SingleMuon_2016H_v2/"""
        """161031_120512/0000/L1Ntuple_999.root""",
        """root://eoscms.cern.ch//eos/cms/store/group/dpg_trigger/"""
        """comm_trigger/L1Trigger/L1Menu2016/Stage2/"""
        """l1t-integration-v88p1-CMSSW-8021/SingleMuon/"""
        """crab_l1t-integration-v88p1-CMSSW-8021__SingleMuon_2016H_v2/"""
        """161031_120512/0000/L1Ntuple_99*.root""",
    ]
    for i, path in enumerate(test_paths):
        expanded = glob(path)
        print(i, path)
        print(i, expanded)
