import unittest
import pandas as pd
import numpy as np
from scipy import stats
from scripts.hypothesis_testing import ABHypothesisTesting

class TestABHypothesisTesting(unittest.TestCase):

    def setUp(self):
        # Create a sample DataFrame for testing
        self.data = pd.DataFrame({
            'Province': ['A', 'A', 'B', 'B', 'C', 'C'],
            'PostalCode': ['123', '123', '456', '456', '789', '789'],
            'Gender': ['Male', 'Female', 'Male', 'Female', 'Male', 'Female'],
            'TotalPremium': [1000, 1500, 2000, 2500, 3000, 3500],
            'TotalClaims': [5, 10, 15, 20, 25, 30]
        })
        self.ab_test = ABHypothesisTesting(self.data)

    def test_segment_data(self):
        # Test segmentation with include and exclude
        result = self.ab_test._segment_data('Province', value='A')
        expected_result = self.data[self.data['Province'] == 'A']
        pd.testing.assert_frame_equal(result, expected_result)

        result = self.ab_test._segment_data('Gender', exclude_values=['Female'])
        expected_result = self.data[self.data['Gender'] == 'Male']
        pd.testing.assert_frame_equal(result, expected_result)

    def test_check_identical_values(self):
        # Test with a column that has all identical values
        # Make sure to modify the sample data as needed
        self.assertFalse(self.ab_test._check_identical_values('TotalPremium'))

        # Test with a column that has different values
        self.assertFalse(self.ab_test._check_identical_values('Gender'))
        
    def test_chi_squared_test(self):
        chi2, p_value = self.ab_test._chi_squared_test('Province', 'TotalPremium')
        self.assertIsInstance(chi2, float)
        self.assertIsInstance(p_value, float)

    def test_t_test(self):
        group_a = self.ab_test._segment_data('Gender', value='Male')
        group_b = self.ab_test._segment_data('Gender', value='Female')
        t_stat, p_value = self.ab_test._t_test(group_a, group_b, 'TotalPremium')
        self.assertIsInstance(t_stat, float)
        self.assertIsInstance(p_value, float)

    def test_z_test(self):
        group_a = self.ab_test._segment_data('PostalCode', value='123')
        group_b = self.ab_test._segment_data('PostalCode', value='456')
        z_stat, p_value = self.ab_test._z_test(group_a, group_b, 'TotalPremium')
        self.assertIsInstance(z_stat, float)
        self.assertIsInstance(p_value, float)

    def test_interpret_p_value(self):
        self.assertEqual(self.ab_test._interpret_p_value(0.01), "Reject the null hypothesis.")
        self.assertEqual(self.ab_test._interpret_p_value(0.1), "Fail to reject the null hypothesis.")
        self.assertEqual(self.ab_test._interpret_p_value(None), "Test skipped due to identical values.")

    def test_run_all_tests(self):
        results = self.ab_test.run_all_tests()
        self.assertIn('Risk Differences Across Provinces', results)
        self.assertIn('Risk Differences Between Postal Codes', results)
        self.assertIn('Margin Differences Between Postal Codes', results)
        self.assertIn('Risk Differences Between Women and Men', results)
        self.assertIsInstance(results['Risk Differences Across Provinces'], str)
        self.assertIsInstance(results['Risk Differences Between Postal Codes'], str)
        self.assertIsInstance(results['Margin Differences Between Postal Codes'], str)
        self.assertIsInstance(results['Risk Differences Between Women and Men'], str)

if __name__ == '__main__':
    unittest.main()