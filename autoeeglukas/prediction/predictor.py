import logging


class Predictor(object):
    """
    """

    def introduce(self):
        logging.info("\t\tHi, I am the predictor! I am part of the pipeline and I take all the available training data "
                     "and use the best training configuration to achieve the best possible score on your test data.")

    def predict(self, cmd_args):
        return

    def __init__(self):
        self.introduce()
