import os
# also check https://stackoverflow.com/questions/41861427/python-3-5-how-to-dynamically-import-a-module-given-the-full-file-path-in-the
os.sys.path.insert(0, '/home/schirrmr/braindecode/code/')
os.sys.path.insert(0, '/home/schirrmr/braindecode/code/braindecode/')
os.sys.path.insert(0, '/home/schirrmr/code/auto-diagnosis/')
import logging
import time

import numpy as np
from numpy.random import RandomState
import resampy
from torch import optim
import torch.nn.functional as F
import torch as th
from tensorboardX import SummaryWriter

from braindecode.datautil.signal_target import SignalAndTarget
from braindecode.torch_ext.util import np_to_var
from braindecode.torch_ext.util import set_random_seeds
from braindecode.experiments.experiment import Experiment
from braindecode.datautil.iterators import CropsFromTrialsIterator
from braindecode.experiments.monitors import (RuntimeMonitor, LossMonitor,
                                              CroppedTrialMisclassMonitor,
                                              MisclassMonitor)
from autodiag.monitors import CroppedDiagnosisMonitor
from braindecode.experiments.stopcriteria import MaxEpochs, Or
from braindecode.datautil.iterators import get_balanced_batches
from braindecode.torch_ext.constraints import MaxNormDefaultConstraint
from braindecode.torch_ext.optimizers import AdamW
from braindecode.torch_ext.schedulers import CosineAnnealing, ScheduledOptimizer
from braindecode.datautil.splitters import concatenate_sets
from copy import copy


from autodiag.dataset import  DiagnosisSet

log = logging.getLogger(__name__)

def dummy():
    print("import worked...")

def create_set(X, y, inds):
    """
    X list and y nparray
    :return: 
    """
    new_X = []
    for i in inds:
        new_X.append(X[i])
    new_y = y[inds]
    return SignalAndTarget(new_X, new_y)


class TrainValidTestSplitter(object):
    def __init__(self, n_folds, i_test_fold, shuffle):
        self.n_folds = n_folds
        self.i_test_fold = i_test_fold
        self.rng = RandomState(39483948)
        self.shuffle = shuffle

    def split(self, X, y,):
        if len(X) < self.n_folds:
            raise ValueError("Less Trials: {:d} than folds: {:d}".format(
                len(X), self.n_folds
            ))
        folds = get_balanced_batches(len(X), self.rng, self.shuffle,
                                     n_batches=self.n_folds)
        test_inds = folds[self.i_test_fold]
        valid_inds = folds[self.i_test_fold - 1]
        all_inds = list(range(len(X)))
        train_inds = np.setdiff1d(all_inds, np.union1d(test_inds, valid_inds))
        assert np.intersect1d(train_inds, valid_inds).size == 0
        assert np.intersect1d(train_inds, test_inds).size == 0
        assert np.intersect1d(valid_inds, test_inds).size == 0
        assert np.array_equal(np.sort(
            np.union1d(train_inds, np.union1d(valid_inds, test_inds))),
            all_inds)

        train_set = create_set(X, y, train_inds)
        valid_set = create_set(X, y, valid_inds)
        test_set = create_set(X, y, test_inds)

        return train_set, valid_set, test_set


class TrainValidSplitter(object):
    def __init__(self, n_folds, i_valid_fold, shuffle):
        self.n_folds = n_folds
        self.i_valid_fold = i_valid_fold
        self.rng = RandomState(39483948)
        self.shuffle = shuffle

    def split(self, X, y):
        if len(X) < self.n_folds:
            raise ValueError("Less Trials: {:d} than folds: {:d}".format(
                len(X), self.n_folds
            ))
        folds = get_balanced_batches(len(X), self.rng, self.shuffle,
                                     n_batches=self.n_folds)
        valid_inds = folds[self.i_valid_fold]
        all_inds = list(range(len(X)))
        train_inds = np.setdiff1d(all_inds, valid_inds)
        assert np.intersect1d(train_inds, valid_inds).size == 0
        assert np.array_equal(np.sort(np.union1d(train_inds, valid_inds)),
            all_inds)

        train_set = create_set(X, y, train_inds)
        valid_set = create_set(X, y, valid_inds)
        return train_set, valid_set

def running_mean(arr, window_len, axis=0):
    # adapted from http://stackoverflow.com/a/27681394/1469195
    # need to pad to get correct first value also
    arr_padded = np.insert(arr,0,values=0,axis=axis)
    cumsum = np.cumsum(arr_padded,axis=axis)
    later_sums = np.take(cumsum, range(window_len, arr_padded.shape[axis]),
        axis=axis)
    earlier_sums = np.take(cumsum, range(0, arr_padded.shape[axis] - window_len),
        axis=axis)

    moving_average = (later_sums - earlier_sums) / float(window_len)
    return moving_average


def padded_moving_mean(arr, axis, n_window):
    """Pads by replicating n_window first and last elements
    and putting them at end and start (no reflection)"""
    start_pad_inds = list(range(0, n_window // 2))
    end_pad_inds = list(range(arr.shape[axis] - (n_window // 2),
                              arr.shape[axis]))
    arr = np.concatenate((arr.take(start_pad_inds, axis=axis),
                          arr,
                          arr.take(end_pad_inds, axis=axis)),
                         axis=axis)
    mov_mean = running_mean(arr, window_len=n_window, axis=axis)
    return mov_mean


def shrink_spikes(example, threshold, axis, n_window):
    """Example could be single example or all...
    should work for both."""
    run_mean = padded_moving_mean(example.astype(np.float32),
        axis=axis, n_window=n_window)
    abs_run_mean = np.abs(run_mean)
    is_relevant = (abs_run_mean > threshold)

    cleaned_example = example - is_relevant * (run_mean - (
             np.sign(run_mean) * (threshold +
            np.log(np.maximum(abs_run_mean - threshold + 1, 0.01)))))
    return cleaned_example


def run_exp(max_recording_mins, n_recordings,
            sec_to_cut, duration_recording_mins, max_abs_val,
            shrink_val,
            sampling_freq,
            divisor,
            n_folds, i_test_fold,
            shuffle,
            merge_train_valid,
            model,
            input_time_length,
            optimizer,
            learning_rate,
            weight_decay,
            scheduler,
            model_constraint,
            batch_size, max_epochs,
            only_return_exp,
            time_cut_off_sec,
            start_time,
            test_on_eval,
            test_recording_mins,
            sensor_types,
            log_dir,
            np_th_seed):
    cuda = True
    import torch.backends.cudnn as cudnn
    cudnn.benchmark = True
    if optimizer == 'adam':
        assert merge_train_valid == False
    else:
        assert optimizer == 'adamw'
        assert merge_train_valid == True


    preproc_functions = []
    preproc_functions.append(
        lambda data, fs: (data[:, int(sec_to_cut * fs):-int(
            sec_to_cut * fs)], fs))
    preproc_functions.append(
        lambda data, fs: (data[:, :int(duration_recording_mins * 60 * fs)], fs))
    if max_abs_val is not None:
        preproc_functions.append(lambda data, fs:
                                 (np.clip(data, -max_abs_val, max_abs_val), fs))
    if shrink_val is not None:
        preproc_functions.append(lambda data, fs:
                                 (shrink_spikes(
                                     data, shrink_val, 1, 9,), fs))

    preproc_functions.append(lambda data, fs: (resampy.resample(data, fs,
                                                                sampling_freq,
                                                                axis=1,
                                                                filter='kaiser_fast'),
                                               sampling_freq))
    if divisor is not None:
        preproc_functions.append(lambda data, fs: (data / divisor, fs))

    dataset = DiagnosisSet(n_recordings=n_recordings,
                        max_recording_mins=max_recording_mins,
                        preproc_functions=preproc_functions,
                           train_or_eval='train',
                           sensor_types=sensor_types)

    if test_on_eval:
        if test_recording_mins is None:
            test_recording_mins = duration_recording_mins
        test_preproc_functions = copy(preproc_functions)
        test_preproc_functions[1] = lambda data, fs: (
            data[:, :int(test_recording_mins * 60 * fs)], fs)
        test_dataset = DiagnosisSet(n_recordings=n_recordings,
                                max_recording_mins=None,
                                preproc_functions=test_preproc_functions,
                                train_or_eval='eval',
                                sensor_types=sensor_types)
    if not only_return_exp:
        X,y = dataset.load()
        max_shape = np.max([list(x.shape) for x in X],
                           axis=0)
        assert max_shape[1] == int(duration_recording_mins *
                                   sampling_freq * 60)
        if test_on_eval:
            test_X, test_y = test_dataset.load()
            max_shape = np.max([list(x.shape) for x in test_X],
                               axis=0)
            assert max_shape[1] == int(test_recording_mins *
                                       sampling_freq * 60)
    if not test_on_eval:
        splitter = TrainValidTestSplitter(n_folds, i_test_fold,
                                          shuffle=shuffle)
    else:
        splitter = TrainValidSplitter(n_folds, i_valid_fold=i_test_fold,
                                          shuffle=shuffle)
    if not only_return_exp:
        if not test_on_eval:
            train_set, valid_set, test_set = splitter.split(X, y)
        else:

            train_set, valid_set = splitter.split(X, y)
            test_set = SignalAndTarget(test_X, test_y)
            del test_X, test_y
        del X,y # shouldn't be necessary, but just to make sure
        if merge_train_valid:
            train_set = concatenate_sets([train_set, valid_set])
            # just reduce valid for faster computations
            valid_set.X = valid_set.X[:8]
            valid_set.y = valid_set.y[:8]
    else:
        train_set = None
        valid_set = None
        test_set = None

    set_random_seeds(seed=np_th_seed, cuda=cuda)

    #optimizer = optim.Adam(model.parameters())
    log.info("Model:\n{:s}".format(str(model)))
    if cuda:
        model.cuda()
    model.eval()
    in_chans = 21
    # determine output size
    test_input = np_to_var(
        np.ones((2, in_chans, input_time_length, 1), dtype=np.float32))
    if cuda:
        test_input = test_input.cuda()
    out = model(test_input)
    n_preds_per_input = out.cpu().data.numpy().shape[2]
    log.info("{:d} predictions per input/trial".format(n_preds_per_input))
    iterator = CropsFromTrialsIterator(batch_size=batch_size,
                                       input_time_length=input_time_length,
                                      n_preds_per_input=n_preds_per_input,
                                       seed=np_th_seed)
    assert optimizer in ['adam', 'adamw'], ("Expect optimizer to be either "
                                            "adam or adamw")
    schedule_weight_decay = optimizer == 'adamw'
    if optimizer == 'adam':
        optim_class = optim.Adam
        assert schedule_weight_decay == False
        assert merge_train_valid == False
    else:
        optim_class = AdamW
        assert schedule_weight_decay == True
        assert merge_train_valid == True

    optimizer = optim_class(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    if scheduler is not None:
        assert scheduler =='cosine'
        n_updates_per_epoch = sum(
            [1 for _ in iterator.get_batches(train_set, shuffle=True)])
        # Adapt if you have a different number of epochs
        n_updates_per_period = n_updates_per_epoch * max_epochs
        scheduler = CosineAnnealing(n_updates_per_period)
        optimizer = ScheduledOptimizer(scheduler, optimizer,
                                       schedule_weight_decay=schedule_weight_decay)
    loss_function = lambda preds, targets: F.nll_loss(
        th.mean(preds, dim=2, keepdim=False), targets)

    if model_constraint is not None:
        assert model_constraint == 'defaultnorm'
        model_constraint = MaxNormDefaultConstraint()
    monitors = [LossMonitor(), MisclassMonitor(col_suffix='sample_misclass'),
                CroppedDiagnosisMonitor(input_time_length, n_preds_per_input),
                RuntimeMonitor(),]

    ## HACK: use stop criterion for tensorboard logging
    class LoggingCriterion(object):
        """
        Fake criterion, actually for logging.

        Parameters
        ----------
        max_epochs: int
        """

        def __init__(self, log_dir):
            self.writer = SummaryWriter(log_dir)

        def should_stop(self, epochs_df):
            iteration = len(epochs_df)
            last_row = dict(epochs_df.iloc[-1])
            for key in last_row:
                val = last_row[key]
                self.writer.add_scalar(key, val, iteration)

            # Keep in mind  epoch 0 without training is also part of dataframe
            return False

    stop_criterion = Or([MaxEpochs(max_epochs), LoggingCriterion(log_dir)])
    batch_modifier = None
    exp = Experiment(model, train_set, valid_set, test_set, iterator,
                     loss_function, optimizer, model_constraint,
                     monitors, stop_criterion,
                     remember_best_column='valid_misclass',
                     run_after_early_stop=True, batch_modifier=batch_modifier,
                     cuda=cuda)
    if not only_return_exp:
        # Until first stop
        exp.setup_training()
        exp.monitor_epoch(exp.datasets)
        exp.print_epoch()
        exp.rememberer.remember_epoch(exp.epochs_df, exp.model,
                                      exp.optimizer)

        exp.iterator.reset_rng()
        while not exp.stop_criterion.should_stop(exp.epochs_df):
            if (time.time() - start_time) > time_cut_off_sec:
                log.info("Ran out of time after {:.2f} sec.".format(
                    time.time() - start_time))
                return exp
            log.info("Still in time after {:.2f} sec.".format(
                    time.time() - start_time))
            exp.run_one_epoch(exp.datasets, remember_best=True)
        if (time.time() - start_time) > time_cut_off_sec:
            log.info("Ran out of time after {:.2f} sec.".format(
                time.time() - start_time))
            return exp
        if not merge_train_valid:
            exp.setup_after_stop_training()
            # Run until second stop
            datasets = exp.datasets
            datasets['train'] = concatenate_sets([datasets['train'],
                                                 datasets['valid']])
            exp.monitor_epoch(datasets)
            exp.print_epoch()

            exp.iterator.reset_rng()
            while not exp.stop_criterion.should_stop(exp.epochs_df):
                if (time.time() - start_time) > time_cut_off_sec:
                    log.info("Ran out of time after {:.2f} sec.".format(
                        time.time() - start_time))
                    return exp
                log.info("Still in time after {:.2f} sec.".format(
                        time.time() - start_time))
                exp.run_one_epoch(datasets, remember_best=False)

    else:
        exp.dataset = dataset
        exp.splitter = splitter
    if test_on_eval:
        exp.test_dataset = test_dataset

    return exp

