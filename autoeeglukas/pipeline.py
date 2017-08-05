import logging

from postprocessing import postprocessor
from preprocessing import preprocessor
from prediction import predictor
from training import trainer

from utils import my_io
from utils import stats


class Pipeline(object):
    """ The pipeline object is a container for all the steps for the machine learning pipeline, i.e. preprocessor,
    trainer, predictor, preprocessor. Moreover, there is a stats object that gathers information about a pipeline run.
    """

# ______________________________________________________________________________________________________________________
    def introduce(self):
        logging.info("\t\tHi, I am the classification pipeline! I read your raw input, process it through all the "
                     "required steps and finally provide a classification score to you.")

# ______________________________________________________________________________________________________________________
    def run(self, cmd_args):
        # TODO: add window counts and feature labels if preprocessing is skipped
        feature_files = cmd_args.input
        window_counts = None
        feature_labels = None
        rec_names = None
        self.stats = stats.Stats()


# --------------------------------------------------- PREPROCESSING ---------------------------------------------------#
        if not cmd_args.no_pre:
            self.preprocessor = preprocessor.Preprocessor(self.stats)
            feature_files, window_counts, self.stats = self.preprocessor.preprocess(cmd_args)
        else:
            logging.info("\t\tPreprocessing is skipped.")

        if not cmd_args.pre_only:
            if cmd_args.no_pre and cmd_args.window != 'boxcar' or cmd_args.windowsize != 2 or cmd_args.overlap != 0 or \
                            cmd_args.bands != '1,4,8,12,18,24,30,60,90':
                logging.warning("\t\tYou cannot change window, window_size, overlap or bands when not doing "
                                "preprocessing. These are feature computation settings. In this mode you are not "
                                "computing new features but using already computed features.")
            cmd_args = my_io.update_cmd_args(cmd_args, feature_files)
            logging.info('\tUpdated the cmd arguments to: {}.'.format(cmd_args))


# -------------------------------------------- TRAINING / CV / AUTOSKLEARN --------------------------------------------#
            self.trainer = trainer.Trainer(jobs=cmd_args.n_proc, auto_skl=cmd_args.auto_skl)
            self.trainer.train(cmd_args, window_counts)
            # TODO: return the classifier, parameters that lead to result that can be used for prediction on test set
            results = self.trainer.get_results()
            (accs, precs, recs, f1s) = results
            logging.info("\t\t\tMean accuracy is {:.2f} ({:.2f}).".format(accs[1], accs[2]))
            if len(cmd_args.input) == 2:
                logging.info("\t\t\tMean precision is {:.2f} ({:.2f}).".format(precs[1], precs[2]))
                logging.info("\t\t\tMean recall is {:.2f} ({:.2f}).".format(recs[1], recs[2]))
                logging.info("\t\t\tMean f1 score is {:.2f} ({:.2f}).".format(f1s[1], f1s[2]))


# ---------------------------------------------------- PREDICTION -----------------------------------------------------#
#             self.predictor = predictor.Predictor()
#             self.predictor.predict(feature_files)


# -------------------------------------------- POSTPROCESSING / ANALYSIS ----------------------------------------------#
            # self.postprocessor = postprocessor.Postprocessor(feature_files, window_counts, rec_names)
            # self.postprocessor.postprocess(cmd_args)
        else:
            logging.info("\t\tTraining, prediction and postprocessing is skipped.")

        if not cmd_args.no_pre:
            self.stats.log()

# ______________________________________________________________________________________________________________________
    def __init__(self):
        self.preprocessor = None
        self.trainer = None
        self.autoskl = None
        self.predictor = None
        self.postprocessor = None
        self.stats = None

        self.introduce()
