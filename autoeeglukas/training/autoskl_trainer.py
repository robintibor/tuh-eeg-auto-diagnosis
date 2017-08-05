import autosklearn
import logging


class AutosklTrainer(object):
    """
    """

# ______________________________________________________________________________________________________________________
    def introduce(self):
        logging.info("\t\tHi, I am the automl trainer!")

# ______________________________________________________________________________________________________________________
    def train(self, feature_files):
        cls = autosklearn.classification.AutoSklearnClassifier()
        cls.fit(x_train, y_train)
        predictions = cls.predict(x_test)