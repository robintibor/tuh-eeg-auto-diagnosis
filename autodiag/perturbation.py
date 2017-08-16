import logging

import numpy as np
from numpy.random import RandomState

from braindecode.datautil.iterators import get_balanced_batches
from braindecode.util import wrap_reshape_apply_fn, corr
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