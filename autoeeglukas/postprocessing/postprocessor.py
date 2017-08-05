import logging

from utils import my_io


class Postprocessor(object):
    """
    """

# ______________________________________________________________________________________________________________________
    def introduce(self):
        logging.info("\tHi, I am the postprocessor. I am part of the pipeline and I try to squeeze the last "
                     "percentages out of your score.")

# ______________________________________________________________________________________________________________________
    def postprocess(self, cmd_args):
        [names, predictions, labels, prob0s, prob1s] = my_io.read_cv_predictions(cmd_args.output)

        for fold_id, fold in enumerate(names):
            classified, misclassified = [], []

            for name_id, name in enumerate(fold):
                curr_prediction, curr_label = predictions[fold_id][name_id], labels[fold_id][name_id]
                # curr_prob0, curr_prob1 = prob0s[fold_id][name_id], prob1s[fold_id][name_id]

                if curr_prediction != curr_label:
                    misclassified.append(name)
                else:
                    classified.append(name)

            print(fold_id)
            print('\n'.join(classified))
            print('\n')
            print('\n'.join(misclassified))
            print('\n\n')

# ______________________________________________________________________________________________________________________
    def __init__(self, feature_files, window_counts, rec_names):
        self.introduce()

        self.feature_analysis = False
        self.electrode_analysis = False
        self.brain_region_analysis = False
