import sys, unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parents[1] / 'src'))
from dataset import split_forgetting_indices
from evaluate import forgetting_score

class UnlearningSmokeTest(unittest.TestCase):
    def test_partition_and_score(self):
        forget, retain = split_forgetting_indices(20, .25, seed=7)
        self.assertEqual(len(forget), 5); self.assertEqual(len(set(forget) & set(retain)), 0)
        self.assertAlmostEqual(forgetting_score(np.array([.9,.8]), np.array([.4,.3])), .5)

if __name__ == '__main__': unittest.main()
