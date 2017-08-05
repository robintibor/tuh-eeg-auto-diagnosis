import numpy as np
import logging


class Stats(object):
    """ This class is a container for all kinds of statistics that can be gathered during a pipeline run """

# ______________________________________________________________________________________________________________________
    def log(self):
        logging.info("\tProcessed {} classes.".format(self.n_classes))

        logging.info("\tFound {} recordings in total.".format(self.n_recordings))

        logging.info("\tThereof {} recordings could not be processed.".format(self.n_corrupted_recordings))

        logging.info("\tThe average duration of recordings is {:.2f}s. The shortest recording is {}s, the longest {}s."
                     .format(np.mean(self.rec_lengths), np.min(self.rec_lengths), np.max(self.rec_lengths)))

        logging.info("\tThe average age of male and female patients is {:.2f} years and {:.2f} years respectively. The "
                     "jungest male patient is {}, the oldest {} years old. The jungest female patient is {}, the oldest"
                     " {} years old. Overall the average age of all patients is {:.2f} years."
                     .format(np.mean(self.ages_m), np.mean(self.ages_f), np.min(self.ages_m), np.max(self.ages_m),
                             np.min(self.ages_f), np.max(self.ages_f), np.mean(self.ages_f + self.ages_m)))

# ______________________________________________________________________________________________________________________
    def __init__(self):
        self.n_classes = 0

        self.n_recordings = 0
        self.n_corrupted_recordings = 0
        self.rec_lengths = []

        self.ages_f = []
        self.ages_m = []
