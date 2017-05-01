import unittest
from rootpy.utils.ext_glob import glob as ext_glob
from glob import glob as py_glob
import os


this_directory = os.path.dirname(os.path.abspath(__file__))


class TestRootGlob(unittest.TestCase):

    def test_local_glob_none(self):
        filename = os.path.join(this_directory, "test_ext_glob.py")
        self.assertEqual([filename], ext_glob(filename))

    def test_local_glob_basename(self):
        filename = os.path.join(this_directory, "*.py")
        self.assertEqual(ext_glob(filename), py_glob(filename))

    def test_local_glob_dirname(self):
        filename = os.path.join(os.path.dirname(this_directory), "*")
        self.assertEqual(ext_glob(filename), py_glob(filename))

    def test_local_glob_both(self):
        filename = os.path.join(os.path.dirname(this_directory), "*", "*.py")
        self.assertEqual(ext_glob(filename), py_glob(filename))

#    def test_xrootd_none(self):
#       filename = """root://eoscms.cern.ch//eos/cms/store/group/"""
#       """dpg_trigger/comm_trigger/L1Trigger/L1Menu2016/Stage2/"""
#       """l1t-integration-v88p1-CMSSW-8021/SingleMuon/"""
#       """crab_l1t-integration-v88p1-CMSSW-8021__SingleMuon_2016H_v2/"""
#       """161031_120512/0000/L1Ntuple_999.root""",

#    def test_xrootd_glob_single(self):
#       filename = """root://eoscms.cern.ch//eos/cms/store/group/"""
#       """dpg_trigger/comm_trigger/L1Trigger/L1Menu2016/Stage2/"""
#       """l1t-integration-v88p1-CMSSW-8021/SingleMuon/"""
#       """crab_l1t-integration-v88p1-CMSSW-8021__SingleMuon_2016H_v2/"""
#       """161031_120512/0000/L1Ntuple_99*.root"""

#    def test_xrootd_glob_multiple(self):
#       filename = """root://eoscms.cern.ch//eos/cms/store/group/"""
#       """dpg_trigger/comm_trigger/L1Trigger/L1Menu2016/Stage2/"""
#       """l1t-integration-v88p1-CMSSW-8021/*/"""
#       """crab_l1t-integration-v88p1-CMSSW-8021__SingleMuon_2016H_v2/"""
#       """161031_1*/0000/L1Ntuple_99*.root"""
