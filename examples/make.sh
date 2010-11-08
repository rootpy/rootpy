#!/bin/bash

root-mkdir data.root

root-cp data_medium_FG.root:D4PD data.root:data

root-cp PythiaAtautauMA120TB20_n2.root:D4PD data.root:PythiaAtautauMA120TB20_n2
root-cp PythiaAtautauMA120TB20_n2.root:D4PDTruth data.root:PythiaAtautauMA120TB20_n2_truth

root-cp PythiaAtautauMA200TB20_n2.root:D4PD data.root:PythiaAtautauMA200TB20_n2
root-cp PythiaAtautauMA200TB20_n2.root:D4PDTruth data.root:PythiaAtautauMA200TB20_n2_truth

root-cp PythiaAtautauMA300TB20_n2.root:D4PD data.root:PythiaAtautauMA300TB20_n2
root-cp PythiaAtautauMA300TB20_n2.root:D4PDTruth data.root:PythiaAtautauMA300TB20_n2_truth

root-cp PythiaWhadtaunu_n2.root:D4PD data.root:PythiaWhadtaunu_n2
root-cp PythiaWhadtaunu_n2.root:D4PDTruth data.root:PythiaWhadtaunu_n2_truth

root-cp PythiaWhadtaunu_n5.root:D4PD data.root:PythiaWhadtaunu_n5
root-cp PythiaWhadtaunu_n5.root:D4PDTruth data.root:PythiaWhadtaunu_n5_truth

root-cp PythiaWhadtaunu.root:D4PD data.root:PythiaWhadtaunu
root-cp PythiaWhadtaunu.root:D4PDTruth data.root:PythiaWhadtaunu_truth

root-cp PythiaZtautau_n2.root:D4PD data.root:PythiaZtautau_n2
root-cp PythiaZtautau_n2.root:D4PDTruth data.root:PythiaZtautau_n2_truth

root-cp PythiaZtautau_n5.root:D4PD data.root:PythiaZtautau_n5
root-cp PythiaZtautau_n5.root:D4PDTruth data.root:PythiaZtautau_n5_truth

root-cp PythiaZtautau.root:D4PD data.root:PythiaZtautau
root-cp PythiaZtautau.root:D4PDTruth data.root:PythiaZtautau_truth
