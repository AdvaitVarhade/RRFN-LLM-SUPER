import unittest
import numpy as np

# A basic test suite for the evaluator logic
from src.evaluator import calculate_rmse, calculate_gini_index, evaluate_novelty

class TestEvaluator(unittest.TestCase):
    def test_calculate_rmse(self):
        y_true = np.array([4.0, 5.0, 2.0, 1.0])
        y_pred = np.array([3.8, 4.9, 2.5, 1.2])
        rmse = calculate_rmse(y_true, y_pred)
        self.assertAlmostEqual(rmse, 0.2915, places=4)

    def test_calculate_gini_index_equal(self):
        # Equal exposure should result in a Gini index of 0
        counts = np.array([10, 10, 10, 10])
        gini = calculate_gini_index(counts)
        self.assertAlmostEqual(gini, 0.0)

    def test_calculate_gini_index_biased(self):
        # Biased exposure should result in a Gini index > 0
        counts = np.array([1000, 10, 2, 0])
        gini = calculate_gini_index(counts)
        self.assertGreater(gini, 0.5)

if __name__ == '__main__':
    unittest.main()
