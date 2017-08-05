import unittest

from testing import trainer_tester
from testing import feature_tester

if __name__ == '__main__':
    suite1 = unittest.TestLoader().loadTestsFromTestCase(trainer_tester.TrainerTester)
    suite2 = unittest.TestLoader().loadTestsFromTestCase(feature_tester.FeatureTester)

    all_tests = unittest.TestSuite([suite1, suite2])
    unittest.TextTestRunner(verbosity=2).run(all_tests)
