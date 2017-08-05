import os
os.sys.path.insert(0, '/home/schirrmr/braindecode/code/')
os.sys.path.insert(0, '/home/schirrmr/braindecode/code/braindecode/')
os.sys.path.insert(0, '/home/schirrmr/code/auto-diagnosis/')
import logging
import time

import numpy as np
import resampy
from torch import optim
import torch.nn.functional as F
import torch as th

from hyperoptim.parse import cartesian_dict_of_lists_product, \
    product_of_list_of_lists_of_dicts
from hyperoptim.util import save_pkl_artifact
from autoeeglukas.utils.my_io import (time_key, read_all_file_names,
                                      get_info_with_mne)
from braindecode.datautil.signalproc import exponential_running_standardize
from braindecode.datautil.signal_target import SignalAndTarget
from braindecode.datautil.splitters import split_into_two_sets
from braindecode.torch_ext.util import np_to_var
from braindecode.torch_ext.util import set_random_seeds
from braindecode.experiments.experiment import Experiment
from braindecode.datautil.iterators import CropsFromTrialsIterator
from braindecode.experiments.monitors import (RuntimeMonitor, LossMonitor,
                                              CroppedTrialMisclassMonitor,
                                              MisclassMonitor)
from braindecode.experiments.stopcriteria import MaxEpochs
from braindecode.models.shallow_fbcsp import ShallowFBCSPNet
from braindecode.models.util import to_dense_prediction_model


log = logging.getLogger(__name__)
log.setLevel('DEBUG')

def get_templates():
    return {}


def get_grid_param_list():
    dictlistprod = cartesian_dict_of_lists_product
    default_params = [{
        'save_folder': './data/models/pytorch/auto-diag/first-try/',
    }]


    grid_params = product_of_list_of_lists_of_dicts([
        default_params,
    ])

    return grid_params


def sample_config_params(rng, params):
    return params

def load_data(fname, preproc_functions):
    cnt, sfreq, n_samples, n_channels, chan_names, n_sec = get_info_with_mne(
        fname)
    log.info("Load data...")
    cnt.load_data()

    wanted_elecs = ['A1', 'A2', 'C3', 'C4', 'CZ', 'F3', 'F4', 'F7', 'F8', 'FP1',
                    'FP2', 'FZ', 'O1', 'O2',
                    'P3', 'P4', 'PZ', 'T3', 'T4', 'T5', 'T6']

    selected_ch_names = []
    for wanted_part in wanted_elecs:
        wanted_found_name = []
        for ch_name in cnt.ch_names:
            if ' ' + wanted_part + '-' in ch_name:
                wanted_found_name.append(ch_name)
        assert len(wanted_found_name) == 1
        selected_ch_names.append(wanted_found_name[0])

    cnt = cnt.pick_channels(selected_ch_names)

    assert np.array_equal(cnt.ch_names, selected_ch_names)
    assert len(cnt.ch_names)  == 21, (
        "Expected 21 channel names, got {:d} channel names".format(
            len(cnt.ch_names)))
    # change from volt to mikrovolt
    data = (cnt.get_data() * 1e6).astype(np.float32)
    fs = cnt.info['sfreq']
    log.info("Preprocessing...")
    for fn in preproc_functions:
        log.info(fn)
        data, fs = fn(data, fs)
        data = data.astype(np.float32)
        fs = float(fs)
    return data


class DiagnosisSet(object):
    def __init__(self, n_recordings, preproc_functions):
        self.n_recordings = n_recordings
        self.preproc_functions = preproc_functions

    def load(self):
        return



def run_exp():
    max_recording_mins = 35
    sampling_freq = 100
    sec_to_cut = 60
    max_abs_val = 800
    n_recordings = 10
    stop_recording_mins = 3
    test_fraction = 0.25
    valid_of_train_fraction = 0.2
    cuda = True
    input_time_length = 1200
    final_conv_length = 20
    batch_size = 64
    max_epochs = 50
    log.info("Read file names")
    normal_file_names = read_all_file_names(
        '/home/gemeinl/data/normal_abnormal/normalv1.1.0/v1.1.0/edf/train/normal/',
        '.edf',
        key='time')
    abnormal_file_names = read_all_file_names(
        '/home/gemeinl/data/normal_abnormal/abnormalv1.1.0/v1.1.0/edf/train/',
        '.edf',
        key='time')

    all_file_names = normal_file_names + abnormal_file_names

    all_file_names = sorted(all_file_names, key=time_key)

    abnorm_counts = [fname.count('abnormal') for fname in all_file_names]
    assert set(abnorm_counts) == set([1, 3])
    labels = np.array(abnorm_counts) == 3
    labels = labels.astype(np.int64)
    log.info("Read recording lengths...")
    lengths = np.load('/home/schirrmr/code/auto-diagnosis/sorted-recording-lengths.npy')

    mask = lengths < max_recording_mins * 60
    cleaned_file_names = np.array(all_file_names)[mask]
    cleaned_labels = labels[mask]

    preproc_functions = []
    preproc_functions.append(
        lambda data, fs: (data[:, int(sec_to_cut * fs):-int(
            sec_to_cut * fs)], fs))
    preproc_functions.append(
        lambda data, fs: (data[:, :int(stop_recording_mins * 60 * fs)], fs))
    #preproc_functions.append(lambda data, fs:
    #                         (np.clip(data, -max_abs_val, max_abs_val), fs))
    preproc_functions.append(lambda data, fs: (resampy.resample(data, fs,
                                                                sampling_freq,
                                                                axis=1,
                                                                filter='kaiser_fast'),
                                               sampling_freq))
    preproc_functions.append(lambda data, fs: (exponential_running_standardize(
        data.T, factor_new=0.001).T, fs))

    #X, y = DiagnosisSet(n_recordings, preproc_functions).load()

    X = []
    y = []
    for i_fname, fname in enumerate(cleaned_file_names[:n_recordings]):
        x = load_data(fname, preproc_functions=preproc_functions)
        assert x is not None
        X.append(x)
        y.append(cleaned_labels[i_fname])
    y = np.array(y)
    n_split = int((1.0 - test_fraction) * len(y))
    train_set = SignalAndTarget(X[:n_split], y=y[:n_split])
    test_set = SignalAndTarget(X[n_split:], y=y[n_split:])
    first_set_fraction = 1 - valid_of_train_fraction
    train_set, valid_set = split_into_two_sets(
        train_set, first_set_fraction=first_set_fraction)
    del X,y # shouldn't be necessary, but just to make sure

    set_random_seeds(seed=20170629, cuda=cuda)
    # This will determine how many crops are processed in parallel
    n_classes = 2
    in_chans = train_set.X[0].shape[0]
    model = ShallowFBCSPNet(in_chans=in_chans, n_classes=n_classes,
                            input_time_length=input_time_length,
                            final_conv_length=final_conv_length).create_network()

    optimizer = optim.Adam(model.parameters())
    to_dense_prediction_model(model)
    log.info("Model:\n{:s}".format(str(model)))
    if cuda:
        model.cuda()
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
                                      n_preds_per_input=n_preds_per_input)
    loss_function = lambda preds, targets: F.nll_loss(th.mean(preds, dim=2)[:,:,0],
                                                      targets)
    model_constraint = None
    monitors = [LossMonitor(), MisclassMonitor(col_suffix='sample_misclass'),
                CroppedTrialMisclassMonitor(input_time_length),
                RuntimeMonitor(),]
    stop_criterion = MaxEpochs(max_epochs)
    exp = Experiment(model, train_set, valid_set, test_set, iterator,
                     loss_function, optimizer, model_constraint,
                     monitors, stop_criterion,
                     remember_best_column='valid_misclass',
                     run_after_early_stop=True, batch_modifier=None, cuda=cuda)

    exp.run()

    return exp

def run(ex, ):
    start_time = time.time()
    ex.info['finished'] = False
    exp = run_exp()
    last_row = exp.epochs_df.iloc[-1]
    end_time = time.time()
    run_time = end_time - start_time
    ex.info['finished'] = True

    for key, val in last_row.iteritems():
        ex.info[key] = float(val)
    ex.info['runtime'] = run_time
    save_pkl_artifact(ex, exp.epochs_df, 'epochs_df.pkl')
    save_pkl_artifact(ex, exp.before_stop_df, 'before_stop_df.pkl')
