"""
Reproduce the standard glob package behaviour but use TSystem to be able to
query remote file systems such as xrootd
"""
from __future__ import print_function
from rootpy.ROOT import gSystem
import glob as gl
import os.path
import fnmatch


__all__ = ["glob", "iglob"]


def __directory_iter(directory):
    while True:
        try:
            file = gSystem.GetDirEntry(directory)
            if not file:
                break
            yield file
        except TypeError:
            break


def glob(pathname):
    # Let normal python glob try first
    try_glob = gl.glob(pathname)
    if try_glob:
        return try_glob

    # If pathname does not contain a wildcard:
    if not gl.has_magic(pathname):
        return [pathname]

    # Else use ROOT's remote system querying
    return root_glob(pathname)


def root_glob(pathname):
    # Split the pathname into a directory and basename
    # (which should include the wild-card)
    dirs, basename = os.path.split(pathname)

    if gl.has_magic(dirs):
        dirs = root_glob(dirs)
    else:
        dirs = [dirs]

    files = []
    for dirname in dirs:
        # Uses `TSystem` to open the directory.
        # TSystem itself wraps up the calls needed to query xrootd.
        dirname = gSystem.ExpandPathName(dirname)
        directory = gSystem.OpenDirectory(dirname)

        if directory:
            for file in __directory_iter(directory):
                if file in [".", ".."]:
                    continue
                if not fnmatch.fnmatchcase(file, basename):
                    continue
                files.append(os.path.join(dirname, file))
            try:
                gSystem.FreeDirectory(directory)
            except TypeError:
                pass
    return files


def iglob(pathname):
    for name in glob(pathname):
        yield name


if __name__ == "__main__":
    test_paths = [
        "*.*",
        "*/*.txt",
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
        """root://eoscms.cern.ch//eos/cms/store/group/dpg_trigger/"""
        """comm_trigger/L1Trigger/L1Menu2016/Stage2/"""
        """l1t-integration-v88p1-CMSSW-8021/SingleMuon/"""
        """crab_l1t-integration-v88p1-CMSSW-8021__SingleMuon_2016H_v*/"""
        """161031_120*/0000/L1Ntuple_99*.root""",
        """root://eoscms.cern.ch//eos/cms/store/group/dpg_trigger/"""
        """comm_trigger/L1Trigger/L1Menu2016/Stage2/"""
        """l1t-integration-v88p1-CMSSW-8021/SingleMuon/"""
        """crab_l1t-integration-v88p1-CMSSW-8021__SingleMuon_2016H_v*/"""
        """161031_120*""",
    ]
    import pprint
    for i, path in enumerate(test_paths):
        print(path, "=>")
        expanded = glob(path)
        print(len(expanded), "files:", pprint.pformat(expanded))
