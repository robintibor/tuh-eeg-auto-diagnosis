import numpy as np
import unittest
import inspect
import sys
import os

cmd_folder = os.path.realpath(os.path.abspath(os.path.split(inspect.getfile(inspect.currentframe()))[0]))
cmd_folder = os.path.realpath(os.path.join(cmd_folder, ".."))
if cmd_folder not in sys.path:
    sys.path.insert(0, cmd_folder)

from training import trainer


class TrainerTester(unittest.TestCase):
    """
    """
# ______________________________________________________________________________________________________________________
    def setUp(self):
        self.trnr = trainer.Trainer()
        self.trnr.folds = 3
        self.a = np.arange(24, dtype=float).reshape(2, 3, 4)
        """ this is unreal feature data. it has 2 classes x 3 recordings x 4 values
        np.array([
            [[0,  1,  2,  3],
             [4,  5,  6,  7],
             [8,  9, 10, 11]],

            [[12, 13, 14, 15],
             [16, 17, 18, 19],
             [20, 21, 22, 23]]
        ])
        """
        # this is needed since without it it comes to confusion when predicting the first fold
        self.a[1] *= 2
        self.train_folds, self.test_folds = self.trnr.create_folds(self.a)

# ______________________________________________________________________________________________________________________
    def test_folds(self):
        self.assertEqual(type(self.train_folds), list)

        # this is for the median over all channels type of feature
        # expecting shape: n_classes=2 x n_folds=3 x n_recordings=2, n_features=4
        expected_train_folds = np.array([
            [[[4,  5,  6,  7],
              [8,  9, 10, 11]],

             [[0, 1, 2, 3],
              [8, 9, 10, 11]],

             [[0, 1, 2, 3],
              [4, 5, 6, 7]]],


            [[[16, 17, 18, 19],
              [20, 21, 22, 23]],

             [[12, 13, 14, 15],
              [20, 21, 22, 23]],

             [[12, 13, 14, 15],
              [16, 17, 18, 19]]]
        ])
        expected_train_folds[1] *= 2
        self.assertEqual(expected_train_folds.shape, np.array(self.train_folds).shape)
        np.testing.assert_array_equal(expected_train_folds, self.train_folds)

        # expecting shape: n_classes=2 x n_folds=3 x n_recordings=1, n_features=4
        expected_test_folds = np.array([
            [[[0, 1, 2, 3]],

             [[4, 5, 6, 7]],

             [[8, 9, 10, 11]]],


            [[[12, 13, 14, 15]],

             [[16, 17, 18, 19]],

             [[20, 21, 22, 23]]]
        ])
        expected_test_folds[1] *= 2
        self.assertEqual(expected_test_folds.shape, np.array(self.test_folds).shape)
        np.testing.assert_array_equal(expected_test_folds, self.test_folds)

# ______________________________________________________________________________________________________________________
    def test_cross_validation(self):
        # in the above example the prediction should be absolutely accurate, tehrefore all the metrics evaluate to 100%
        self.trnr.cross_validate(self.a, None, self.train_folds, self.test_folds)
        expected_accs = [100., 100., 100.]
        self.assertEqual(expected_accs, self.trnr.accs)

        expected_precs = [100., 100., 100.]
        self.assertEqual(expected_precs, self.trnr.precs)

        expected_recs = [100., 100., 100.]
        self.assertEqual(expected_recs, self.trnr.recs)

        expected_f1s = [100., 100., 100.]
        self.assertEqual(expected_f1s, self.trnr.f1s)

# ______________________________________________________________________________________________________________________
    def test_get_train_test(self):
        expected_train_x = [
            np.array([
                [4, 5, 6, 7],
                [8, 9, 10, 11],
                [32, 34, 36, 38],
                [40, 42, 44, 46]
            ]),
            np.array([
                [0, 1, 2, 3],
                [8, 9, 10, 11],
                [24, 26, 28, 30],
                [40, 42, 44, 46]
            ]),
            np.array([
                [0, 1, 2, 3],
                [4, 5, 6, 7],
                [24, 26, 28, 30],
                [32, 34, 36, 38]
            ])
        ]
        expected_test_x = [
            np.array([
                [0, 1, 2, 3],
                [24, 26, 28, 30]
            ]),
            np.array([
                [4, 5, 6, 7],
                [32, 34, 36, 38]
            ]),
            np.array([
                [8, 9, 10, 11],
                [40, 42, 44, 46]
            ])
        ]

        for fold in range(self.trnr.folds):
            train_x, train_y, test_x, test_y = self.trnr.get_train_test(fold, self.a, self.train_folds, self.test_folds)
            # check if the assembled folds are correct
            np.testing.assert_array_equal(train_x, expected_train_x[fold])
            np.testing.assert_array_equal(test_x, expected_test_x[fold])

            # check if the labels are correct
            self.assertEqual(train_y, len(self.train_folds[0][fold]) * [0] + len(self.train_folds[0][fold]) * [1])
            self.assertEqual(test_y, len(self.test_folds[0][fold]) * [0] + len(self.test_folds[0][fold]) * [1])

# ______________________________________________________________________________________________________________________
if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TrainerTester)
    unittest.TextTestRunner(verbosity=2).run(suite)
