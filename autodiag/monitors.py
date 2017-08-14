import numpy as np

from braindecode.datautil.iterators import _compute_start_stop_block_inds


class CroppedNonDenseTrialMisclassMonitor(object):
    """
    Compute trialwise misclasses from predictions for crops for non-dense predictions.

    Parameters
    ----------
    input_time_length: int
        Temporal length of one input to the model.
    """

    def __init__(self, input_time_length, n_preds_per_input):
        self.input_time_length = input_time_length
        self.n_preds_per_input = n_preds_per_input

    def monitor_epoch(self, ):
        return

    def monitor_set(self, setname, all_preds, all_losses,
                    all_batch_sizes, all_targets, dataset):
        """Assuming one hot encoding for now"""
        n_trials = len(dataset.X)
        i_pred_starts = [self.input_time_length -
                         self.n_preds_per_input] * n_trials
        i_pred_stops = [t.shape[1] for t in dataset.X]

        start_stop_block_inds_per_trial = _compute_start_stop_block_inds(
            i_pred_starts,
            i_pred_stops, self.input_time_length, self.n_preds_per_input,
            False)

        n_rows_per_trial = [len(block_inds) for block_inds in
                            start_stop_block_inds_per_trial]

        all_preds_arr = np.concatenate(all_preds, axis=0)
        i_row = 0
        preds_per_trial = []
        for n_rows in n_rows_per_trial:
            preds_per_trial.append(all_preds_arr[i_row:i_row + n_rows])
            i_row += n_rows

        mean_preds_per_trial = [np.mean(preds, axis=(0, 2)) for preds in
                                preds_per_trial]
        mean_preds_per_trial = np.array(mean_preds_per_trial)

        pred_labels_per_trial = np.argmax(mean_preds_per_trial, axis=1)
        assert pred_labels_per_trial.shape == dataset.y.shape
        accuracy = np.mean(pred_labels_per_trial == dataset.y)
        misclass = 1 - accuracy
        column_name = "{:s}_misclass".format(setname)
        return {column_name: float(misclass)}