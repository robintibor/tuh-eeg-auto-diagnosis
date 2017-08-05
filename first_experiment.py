import os
import site
import sys
os.sys.path.insert(0, '/home/schirrmr/braindecode/code/')
os.sys.path.insert(0, '/home/schirrmr/braindecode/code/braindecode/')
os.sys.path.insert(0, '/home/schirrmr/code/auto-diagnosis/')
import numpy as np
from autoeeglukas.utils.my_io import (time_key, read_all_file_names,
                                      get_info_with_mne)
import logging
log = logging.getLogger(__name__)
log.setLevel('DEBUG')
logging.basicConfig(format='%(asctime)s %(levelname)s : %(message)s',
                     level=logging.DEBUG, stream=sys.stdout)

normal_file_names = read_all_file_names('/home/gemeinl/data/normal_abnormal/normalv1.1.0/v1.1.0/edf/train/normal/', '.edf',
                    key='time')
abnormal_file_names = read_all_file_names('/home/gemeinl/data/normal_abnormal/abnormalv1.1.0/v1.1.0/edf/train/', '.edf',
                    key='time')

all_file_names = normal_file_names + abnormal_file_names

all_file_names = sorted(all_file_names,key=time_key)

abnorm_counts = [fname.count('abnormal') for fname in all_file_names]
assert set(abnorm_counts) == set([1,3])
labels = np.array(abnorm_counts) == 3
labels = labels.astype(np.int64)

log.info("Read recording lengths...")
lengths = np.load('/home/schirrmr/code/auto-diagnosis/sorted-recording-lengths.npy')
#lengths = [get_recording_length(fname) for fname in all_file_names]

mask = lengths < 35 * 60
cleaned_file_names = np.array(all_file_names)[mask]
cleaned_labels = labels[mask]
print("len cleaned", len(cleaned_labels))

import resampy
from braindecode.datautil.signalproc import exponential_running_standardize


def load_data(fname, sec_to_cut=20, sampling_freq=100):
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
    data = cnt.get_data().astype(np.float32)
    data = data[:, int(sec_to_cut * cnt.info['sfreq']):-int(
        sec_to_cut * cnt.info['sfreq'])]

    if sampling_freq != cnt.info['sfreq']:
        log.info("Resampling...")
        data = resampy.resample(data.T, cnt.info['sfreq'],
                                sampling_freq, axis=0, filter='kaiser_fast').T
    data = data.astype(np.float32)
    log.info("Exponential standardization...")
    data = exponential_running_standardize(data.T, factor_new=0.001).T
    data = data.astype(np.float32)
    assert data.dtype == 'float32'
    return data

n_recordings = None#2666
X = []
y = []
for i_fname, fname in enumerate(cleaned_file_names[:n_recordings]):
    x = load_data(fname)
    if x is not None:
        X.append(x)
        y.append(cleaned_labels[i_fname])
y = np.array(y)

from braindecode.datautil.signal_target import SignalAndTarget
from braindecode.datautil.splitters import split_into_two_sets

n_split = int(0.75 * len(y))
train_set = SignalAndTarget(X[:n_split], y=y[:n_split])
test_set = SignalAndTarget(X[n_split:], y=y[n_split:])

train_set, valid_set = split_into_two_sets(train_set, first_set_fraction=0.8)

from braindecode.models.shallow_fbcsp import ShallowFBCSPNet
from braindecode.models.deep4 import Deep4Net
from torch import nn
from braindecode.torch_ext.util import set_random_seeds
from braindecode.models.util import to_dense_prediction_model

# Set if you want to use GPU
# You can also use torch.cuda.is_available() to determine if cuda is available on your machine.
cuda = True
set_random_seeds(seed=20170629, cuda=cuda)

# This will determine how many crops are processed in parallel
input_time_length = 1200
n_classes = 2
in_chans = train_set.X[0].shape[0]
# final_conv_length determines the size of the receptive field of the ConvNet
model = ShallowFBCSPNet(in_chans=in_chans, n_classes=n_classes, input_time_length=input_time_length,
                        final_conv_length=20).create_network()
#model = Deep4Net(in_chans, n_classes, input_time_length=input_time_length,
#                            final_conv_length=2).create_network()
to_dense_prediction_model(model)
log.info("Model:\n{:s}".format(str(model)))
if cuda:
    model.cuda()

from torch import optim

optimizer = optim.Adam(model.parameters())

from braindecode.torch_ext.util import np_to_var
# determine output size
test_input = np_to_var(np.ones((2, in_chans, input_time_length, 1), dtype=np.float32))
if cuda:
    test_input = test_input.cuda()
out = model(test_input)
n_preds_per_input = out.cpu().data.numpy().shape[2]
log.info("{:d} predictions per input/trial".format(n_preds_per_input))

from braindecode.experiments.experiment import Experiment
from braindecode.datautil.iterators import CropsFromTrialsIterator
from braindecode.experiments.monitors import RuntimeMonitor, LossMonitor, CroppedTrialMisclassMonitor, MisclassMonitor
from braindecode.experiments.stopcriteria import MaxEpochs
import torch.nn.functional as F
import torch as th
from braindecode.torch_ext.modules import Expression
# Iterator is used to iterate over datasets both for training
# and evaluation
iterator = CropsFromTrialsIterator(batch_size=32,input_time_length=input_time_length,
                                  n_preds_per_input=n_preds_per_input)

# Loss function takes predictions as they come out of the network and the targets
# and returns a loss
loss_function = lambda preds, targets: F.nll_loss(th.mean(preds, dim=2)[:,:,0], targets)

# Could be used to apply some constraint on the models, then should be object
# with apply method that accepts a module
model_constraint = None
# Monitors log the training progress
monitors = [LossMonitor(), MisclassMonitor(col_suffix='sample_misclass'),
            CroppedTrialMisclassMonitor(input_time_length), RuntimeMonitor(),]
# Stop criterion determines when the first stop happens
stop_criterion = MaxEpochs(50)
exp = Experiment(model, train_set, valid_set, test_set, iterator, loss_function, optimizer, model_constraint,
          monitors, stop_criterion, remember_best_column='valid_misclass',
          run_after_early_stop=True, batch_modifier=None, cuda=cuda)

exp.run()

import pickle
pickle.dump(exp.epochs_df, open('shallow-full-set.df.pkl', 'wb'))