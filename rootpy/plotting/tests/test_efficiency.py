# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License

from rootpy.plotting import Efficiency, Hist
import numpy as np
from nose.tools import (raises, assert_equal, assert_almost_equal,
                        assert_raises, assert_true, assert_false)



def test_overall_efficiency():

    for stat_op in range(0, 8):
        #stat_op = 1
        Eff = Efficiency(Hist(20, -3, 3), Hist(20, -3, 3))
        Eff_1bin = Efficiency(Hist(1, -3, 3), Hist(1, -3, 3))
        Eff.SetStatisticOption(stat_op)
        Eff_1bin.SetStatisticOption(stat_op)
    
        NGEN = int(1e3)
        weights = np.random.uniform(0, 1, NGEN)
        values = np.random.normal(0, 3.6, NGEN)
    
        for x, w in zip(values, weights):
            passed = w > 0.5
            Eff.Fill(passed, x)
            Eff_1bin.Fill(passed, x)

        assert_almost_equal(Eff.overall_eff(overflow=True), 
                            Eff_1bin.overall_eff(overflow=True))
        assert_almost_equal(Eff.overall_errors(overflow=True)[0], 
                            Eff_1bin.overall_errors(overflow=True)[0])
        assert_almost_equal(Eff.overall_errors(overflow=True)[1], 
                            Eff_1bin.overall_errors(overflow=True)[1])



if __name__ == "__main__":
    import nose
    nose.runmodule()
    #test_overall_efficiency()
