import numpy as np
import unittest
import inspect
import sys
import os

cmd_folder = os.path.realpath(os.path.abspath(os.path.split(inspect.getfile(inspect.currentframe()))[0]))
cmd_folder = os.path.realpath(os.path.join(cmd_folder, ".."))
if cmd_folder not in sys.path:
    sys.path.insert(0, cmd_folder)

from feature_generation import feature_generator
from preprocessing import recording


class FeatureTester(unittest.TestCase):
    """
    """
# ______________________________________________________________________________________________________________________
    def setUp(self):
        self.feature_generator = feature_generator.FeatureGenerator()

        a = np.array([0, 1, 2, 3])
        rec = recording.Recording("", "", None, 1, 4, 1, "", 4)
        rec.signals = a
        rec.signals_ft = a
        # features = self.feature_generator.generate_features(rec)

# ______________________________________________________________________________________________________________________
    def test_rolling_to_windows(self):
        # self.feature_generator.rolling_to_windows(rolling_feature=np.arange(24).reshape(3, 8), window_size=2)
        pass

# ______________________________________________________________________________________________________________________
    def test_generate_features(self):
        pass

# ______________________________________________________________________________________________________________________
if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(FeatureTester)
    unittest.TextTestRunner(verbosity=2).run(suite)
