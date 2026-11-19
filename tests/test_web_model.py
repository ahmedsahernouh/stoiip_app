import unittest
import numpy as np
from web_model import contact_totals, simulate

class MonteCarloTests(unittest.TestCase):
    def test_zero_uncertainty_reproduces_base_exactly(self):
        base=dict(thickness=100,porosity=.2,saturation=.7,ntg=.8,bo=1.2)
        unc={k:0 for k in [*base,'area']}
        draws,sensitivity=simulate(724080000,base,unc,100,123)
        np.testing.assert_array_equal(draws,np.full(100,724080000.0))
        self.assertEqual(sum(sensitivity.values()),0)

    def test_contact_integrates_volume_not_node_counts(self):
        structure=np.array([[100.,200.],[300.,np.nan]])
        volume=np.array([[10.,50.],[100.,np.nan]])
        contacts=np.array([50.,100.,250.,400.])
        np.testing.assert_array_equal(contact_totals(structure,volume,contacts,True),[0,10,60,160])
        np.testing.assert_array_equal(contact_totals(structure,volume,contacts,False),[160,160,100,0])

    def test_seed_reproducibility_and_finite_samples(self):
        base=dict(thickness=100,porosity=.2,saturation=.7,ntg=.8,bo=1.2)
        unc={k:20 for k in [*base,'area']}
        a,s=simulate(724080000,base,unc,1000,123)
        b,_=simulate(724080000,base,unc,1000,123)
        np.testing.assert_array_equal(a,b)
        self.assertTrue(np.all(np.isfinite(a)&(a>=0)))
        self.assertAlmostEqual(sum(s.values()),1)

if __name__=='__main__':unittest.main()
