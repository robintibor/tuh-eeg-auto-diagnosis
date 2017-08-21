import logging

import numpy as np
from numpy.random import RandomState

from braindecode.datautil.iterators import get_balanced_batches
from braindecode.util import wrap_reshape_apply_fn, corr, cov
from braindecode.visualization.perturbation import gaussian_perturbation

log = logging.getLogger(__name__)


def std_scaled_gaussian_perturbation(amps, rng, factor=0.01):
    """
    Create gaussian noise tensor with same shape as amplitudes.

    Parameters
    ----------
    amps: ndarray
        Amplitudes.
    rng: RandomState
        Random generator.

    Returns
    -------
    perturbation: ndarray
        Perturbations to add to the amplitudes.
    """
    factor = np.float32(factor)
    stds = np.std(amps, axis=0, keepdims=True).astype(np.float32)
    perturbation = rng.randn(*amps.shape).astype(np.float32)
    perturbation = perturbation * stds * factor
    return perturbation


def compute_amplitude_prediction_correlations(pred_fn, examples, n_iterations,
                                              perturb_fn=gaussian_perturbation,
                                              batch_size=30,
                                              seed=((2017, 7, 10)),
                                              original_y=None,
                                             ):
    """
    Perturb input amplitudes and compute correlation between amplitude
    perturbations and prediction changes when pushing perturbed input through
    the prediction function.

    For more details, see [EEGDeepLearning]_.

    Parameters
    ----------
    pred_fn: function
        Function accepting an numpy input and returning prediction.
    examples: ndarray
        Numpy examples, first axis should be example axis.
    n_iterations: int
        Number of iterations to compute.
    perturb_fn: function, optional
        Function accepting amplitude array and random generator and returning
        perturbation. Default is Gaussian perturbation.
    batch_size: int, optional
        Batch size for computing predictions.
    seed: int, optional
        Random generator seed

    Returns
    -------
    amplitude_pred_corrs: ndarray
        Correlations between amplitude perturbations and prediction changes
        for all sensors and frequency bins.

    References
    ----------

    .. [EEGDeepLearning] Schirrmeister, R. T., Springenberg, J. T., Fiederer, L. D. J.,
       Glasstetter, M., Eggensperger, K., Tangermann, M., ... & Ball, T. (2017).
       Deep learning with convolutional neural networks for EEG decoding and
       visualization.
       arXiv preprint arXiv:1703.05051.
    """
    inds_per_batch = get_balanced_batches(
        n_trials=len(examples), rng=None, shuffle=False, batch_size=batch_size)
    log.info("Compute original predictions...")
    orig_preds = [pred_fn(examples[example_inds])
                  for example_inds in inds_per_batch]
    orig_preds_arr = np.concatenate(orig_preds)
    if original_y is not None:
        orig_pred_labels = np.argmax(orig_preds_arr, axis=1)
        orig_accuracy = np.mean(orig_pred_labels == original_y)
        log.info("Original accuracy: {:.2f}...".format(orig_accuracy))
    rng = RandomState(seed)
    fft_input = np.fft.rfft(examples, axis=2).astype(np.complex64)
    amps = np.abs(fft_input).astype(np.float32)
    phases = np.angle(fft_input).astype(np.float32)
    del fft_input

    amp_pred_corrs = []
    new_accuracies = []
    for i_iteration in range(n_iterations):
        log.info("Iteration {:d}...".format(i_iteration))
        log.info("Sample perturbation...")
        perturbation = perturb_fn(amps, rng).astype(np.float32)
        log.info("Compute new amplitudes...")
        # do not allow perturbation to make amplitudes go below
        # zero
        perturbation = np.maximum(-amps, perturbation)
        new_amps = amps + perturbation
        new_amps = new_amps.astype(np.float32)
        log.info("Compute new  complex inputs...")
        new_complex = _amplitude_phase_to_complex(new_amps, phases).astype(np.complex64)
        log.info("Compute new real inputs...")
        new_in = np.fft.irfft(new_complex, axis=2).astype(np.float32)
        del new_complex, new_amps
        log.info("Compute new predictions...")
        new_preds = [pred_fn(new_in[example_inds])
                     for example_inds in inds_per_batch]

        new_preds_arr = np.concatenate(new_preds)
        if original_y is not None:
            new_pred_labels = np.argmax(new_preds_arr, axis=1)
            new_accuracy = np.mean(new_pred_labels == original_y)
            log.info("New accuracy: {:.2f}...".format(new_accuracy))
            new_accuracies.append(new_accuracy)
        diff_preds = new_preds_arr - orig_preds_arr

        log.info("Compute correlation...")
        amp_pred_corr = wrap_reshape_apply_fn(corr, perturbation[:, :, :, 0],
                                              diff_preds,
                                              axis_a=(0,), axis_b=(0))
        amp_pred_corrs.append(amp_pred_corr)
    if original_y is not None:
        return amp_pred_corrs, orig_accuracy, new_accuracies
    else:
        return amp_pred_corrs


def _amplitude_phase_to_complex(amplitude, phase):
    return amplitude * np.cos(phase) + amplitude * np.sin(phase) * np.complex64(1j)


def combine_covs(cov_1, n_1, mean_1_a, mean_1_b, cov_2, n_2, mean_2_a, mean_2_b):
    mean_diff_product = np.dot(np.expand_dims((mean_1_a - mean_2_a), axis=-1), (mean_1_b - mean_2_b)[None])
    return ((n_1-1)*cov_1 + (n_2-1)*cov_2 + mean_diff_product *
            ((n_1*n_2)/float(n_1+n_2))) / float(n_1+n_2-1)

def combine_vars(var_1, n_1, mean_1, var_2, n_2, mean_2):
    #https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    diff = mean_2 - mean_1
    new_m = var_1 * (n_1 - 1) + var_2 * (n_2 - 1) + (diff ** 2) * (n_1 * n_2 / float(n_1 + n_2))
    return new_m / (n_1 +  n_2 - 1)


def compute_amplitude_prediction_correlations_batchwise(
        pred_fn, examples, n_iterations, perturb_fn=gaussian_perturbation,
        batch_size=30, seed=((2017, 7, 10)), original_y=None,):
    """
    Perturb input amplitudes and compute correlation between amplitude
    perturbations and prediction changes when pushing perturbed input through
    the prediction function.

    For more details, see [EEGDeepLearning]_.

    Parameters
    ----------
    pred_fn: function
        Function accepting an numpy input and returning prediction.
    examples: ndarray
        Numpy examples, first axis should be example axis.
    n_iterations: int
        Number of iterations to compute.
    perturb_fn: function, optional
        Function accepting amplitude array and random generator and returning
        perturbation. Default is Gaussian perturbation.
    batch_size: int, optional
        Batch size for computing predictions.
    seed: int, optional
        Random generator seed

    Returns
    -------
    amplitude_pred_corrs: ndarray
        Correlations between amplitude perturbations and prediction changes
        for all sensors and frequency bins.

    References
    ----------

    .. [EEGDeepLearning] Schirrmeister, R. T., Springenberg, J. T., Fiederer, L. D. J.,
       Glasstetter, M., Eggensperger, K., Tangermann, M., ... & Ball, T. (2017).
       Deep learning with convolutional neural networks for EEG decoding and
       visualization.
       arXiv preprint arXiv:1703.05051.
    """
    inds_per_batch = get_balanced_batches(
        n_trials=len(examples), rng=None, shuffle=False, batch_size=batch_size)
    log.info("Compute original predictions...")
    orig_preds = [pred_fn(examples[example_inds])
                  for example_inds in inds_per_batch]
    orig_preds_arr = np.concatenate(orig_preds)
    if original_y is not None:
        orig_pred_labels = np.argmax(orig_preds_arr, axis=1)
        orig_accuracy = np.mean(orig_pred_labels == original_y)
        log.info("Original accuracy: {:.2f}...".format(orig_accuracy))
    amp_pred_corrs = []
    new_accuracies = []
    rng = RandomState(seed)
    for i_iteration in range(n_iterations):
        log.info("Iteration {:d}...".format(i_iteration))
        size_so_far = 0
        mean_perturb_so_far = None
        mean_pred_diff_so_far = None
        var_perturb_so_far = None
        var_pred_diff_so_far = None
        covariance_so_far = None
        all_new_pred_labels = []
        for example_inds in inds_per_batch:
            this_orig_preds = orig_preds_arr[example_inds]
            this_examples = examples[example_inds]
            fft_input = np.fft.rfft(this_examples, axis=2).astype(np.complex64)
            amps = np.abs(fft_input).astype(np.float32)
            phases = np.angle(fft_input).astype(np.float32)
            #log.info("Sample perturbation...")
            perturbation = perturb_fn(amps, rng).astype(np.float32)
            #log.info("Compute new amplitudes...")
            # do not allow perturbation to make amplitudes go below
            # zero
            perturbation = np.maximum(-amps, perturbation)
            new_amps = amps + perturbation
            new_amps = new_amps.astype(np.float32)
            #log.info("Compute new  complex inputs...")
            new_complex = _amplitude_phase_to_complex(new_amps, phases).astype(
                np.complex64)
            #log.info("Compute new real inputs...")
            new_in = np.fft.irfft(new_complex, axis=2).astype(np.float32)
            #log.info("Compute new predictions...")
            new_preds_arr = pred_fn(new_in)
            if original_y is not None:
                new_pred_labels = np.argmax(new_preds_arr, axis=1)
                all_new_pred_labels.append(new_pred_labels)
            diff_preds = new_preds_arr - this_orig_preds
            this_amp_pred_cov = wrap_reshape_apply_fn(cov,
                                                      perturbation[:, :, :, 0],
                                                      diff_preds,
                                                      axis_a=(0,), axis_b=(0))
            var_perturb = np.var(perturbation, axis=0, ddof=1)
            var_pred_diff = np.var(diff_preds, axis=0, ddof=1)
            mean_perturb = np.mean(perturbation, axis=0)
            mean_diff_pred = np.mean(diff_preds)
            if mean_perturb_so_far is None:
                mean_perturb_so_far = mean_perturb
                mean_pred_diff_so_far = mean_diff_pred
                covariance_so_far = this_amp_pred_cov
                var_perturb_so_far = var_perturb
                var_pred_diff_so_far = var_pred_diff
            else:
                covariance_so_far = combine_covs(
                    covariance_so_far, size_so_far,
                    mean_perturb_so_far, mean_pred_diff_so_far,
                    this_amp_pred_cov, len(example_inds),
                    mean_perturb, mean_diff_pred)
                var_perturb_so_far = combine_vars(
                    var_perturb_so_far, size_so_far,
                    mean_perturb_so_far,
                    var_perturb, len(example_inds),
                    mean_perturb, )
                var_pred_diff_so_far = combine_vars(
                    var_pred_diff_so_far, size_so_far,
                    mean_pred_diff_so_far,
                    var_pred_diff, len(example_inds),
                    mean_diff_pred, )
                next_size = size_so_far + len(example_inds)
                mean_perturb_so_far = ((
                    mean_perturb_so_far * size_so_far / float(next_size)) +
                    (mean_perturb * len(example_inds) / float(next_size)))
                mean_pred_diff_so_far = ((
                    mean_pred_diff_so_far * size_so_far / float(next_size)) +
                    (mean_diff_pred * len(example_inds) / float(next_size)))

            size_so_far += len(example_inds)

        all_new_pred_labels = np.concatenate(all_new_pred_labels)
        new_accuracy = np.mean(all_new_pred_labels == original_y)
        assert len(original_y) == len(all_new_pred_labels)
        log.info("New accuracy: {:.2f}...".format(new_accuracy))
        new_accuracies.append(new_accuracy)
        divisor = np.outer(var_perturb_so_far, var_pred_diff_so_far).reshape(
            *var_perturb_so_far.shape + var_pred_diff_so_far.shape).squeeze()
        this_amp_pred_corr = covariance_so_far / divisor
        amp_pred_corrs.append(this_amp_pred_corr)
    if original_y is not None:
        return amp_pred_corrs, orig_accuracy, new_accuracies
    else:
        return amp_pred_corrs


