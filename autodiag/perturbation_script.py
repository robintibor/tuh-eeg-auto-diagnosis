import os
os.sys.path.insert(0, '/home/schirrmr/code/auto-diagnosis/')
import os.path
import logging
import sys

import numpy as np
import torch as th
from torch import nn

from hyperoptim.rerun import rerun_exp
from braindecode.torch_ext.util import var_to_np, np_to_var
from autodiag.perturbation import (compute_amplitude_prediction_correlations,
    std_scaled_gaussian_perturbation,
    compute_amplitude_prediction_correlations_batchwise)
from braindecode.visualization.perturbation import gaussian_perturbation

log = logging.getLogger(__name__)

def fix_file_for_load_exp(folder):
    # possible fix for content
    filename = os.path.join(folder, 'exp_file.py')
    content = open(filename, 'r').read()

    if (not "save_torch_artifact(ex, exp.model.state_dict(), 'model_params.pkl')\n    return exp" in content) and(
        not "else:\n        return exp"
    ):
        "save_torch_artifact(ex, exp.model.state_dict(), 'model_params.pkl')\n\n" in content
        content = content.replace(
            "save_torch_artifact(ex, exp.model.state_dict(), 'model_params.pkl')\n",
            "save_torch_artifact(ex, exp.model.state_dict(), 'model_params.pkl')\n    return exp")

    open(filename, 'w').write(content)


def run_save_for_gaussian(
        pred_fn, train_X_batches, n_iterations, train_y_batches, folder):
    amp_pred_corrs, orig_acc, new_accs = (
        compute_amplitude_prediction_correlations_batchwise(
            pred_fn, train_X_batches,  n_iterations=n_iterations, batch_size=32,
            perturb_fn=gaussian_perturbation,
            original_y=train_y_batches))

    save_filename = os.path.join(folder, 'gaussian.perturbation.{:d}.npy'.format(
        n_iterations
    ))
    log.info("Saving to {:s}".format(save_filename))
    np.save(save_filename, [amp_pred_corrs, orig_acc, new_accs])


def run_save_for_scaled(
        pred_fn, train_X_batches, n_iterations, train_y_batches, folder):
    assert False
    amp_pred_corrs, orig_acc, new_accs = compute_amplitude_prediction_correlations(
        pred_fn, train_X_batches,  n_iterations=n_iterations, batch_size=32,
        perturb_fn=std_scaled_gaussian_perturbation,
        original_y=train_y_batches)

    save_filename = os.path.join(folder, 'scaled.perturbation.{:d}.npy'.format(
        n_iterations
    ))
    log.info("Saving to {:s}".format(save_filename))
    np.save(save_filename, [amp_pred_corrs, orig_acc, new_accs])


if __name__ == '__main__':

    folder = sys.argv[1]
    # Example call:
    # python3.5 perturbation_script.py  /home/schirrmr/data/models/pytorch/auto-diag/10-fold/138/
    #/home/schirrmr/data/models/pytorch/auto-diag/10-fold/138/
    #169
    n_recordings = None#20 # only if I want to limit it
    n_iterations = 30#4#30

    # possible fix if load exp not possible otherwise
    #fix_file_for_load_exp(folder)

    ex = rerun_exp(folder, update_params=dict(only_return_exp=True),
                   save_experiment=False)
    exp = ex.result
    if n_recordings is not None:
        exp.dataset.n_recordings = n_recordings

    X, y = exp.dataset.load()

    train_set, valid_set, test_set = exp.splitter.split(X, y)


    exp.model.load_state_dict(th.load(os.path.join(folder, 'model_params.pkl')))
    exp.model.eval();

    train_batches = list(exp.iterator.get_batches(train_set, shuffle=False))
    train_X_batches = np.concatenate(list(zip(*train_batches))[0]).astype(np.float32)
    train_y_batches = np.concatenate(list(zip(*train_batches))[1])
    del X,y
    del train_set, valid_set, test_set

    model_without_softmax = nn.Sequential()
    for name, module in exp.model.named_children():
        if name == 'softmax':
            break
        model_without_softmax.add_module(name, module)

    pred_fn = lambda x: var_to_np(
        th.mean(model_without_softmax(np_to_var(x).cuda()), dim=2)[:, :, 0, 0])

    log.info("Gaussian perturbation...")
    run_save_for_gaussian(
        pred_fn, train_X_batches, n_iterations, train_y_batches, folder)
    #log.info("Scaled (gaussian) perturbation...")
    #run_save_for_scaled(
    #    pred_fn, train_X_batches, n_iterations, train_y_batches, folder)

