import logging
import numpy as np

from autoeeglukas.utils.my_io import (time_key, read_all_file_names,
                                      get_info_with_mne)

log = logging.getLogger(__name__)

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

def get_all_sorted_file_names_and_labels(train_or_eval):
    normal_path = ('/home/gemeinl/data/normal_abnormal/normalv1.1.0/'
        'v1.1.0/edf/{:s}/normal/'.format(train_or_eval))
    normal_file_names = read_all_file_names(normal_path, '.edf', key='time')
    abnormal_path = ('/home/gemeinl/data/normal_abnormal/abnormalv1.1.0/'
        'v1.1.0/edf/{:s}/abnormal/'.format(train_or_eval))
    abnormal_file_names = read_all_file_names(abnormal_path, '.edf', key='time')

    all_file_names = normal_file_names + abnormal_file_names

    all_file_names = sorted(all_file_names, key=time_key)

    abnorm_counts = [fname.count('abnormal') for fname in all_file_names]
    assert set(abnorm_counts) == set([1, 3])
    labels = np.array(abnorm_counts) == 3
    labels = labels.astype(np.int64)
    return all_file_names, labels

class DiagnosisSet(object):
    def __init__(self, n_recordings, max_recording_mins, preproc_functions,
                 train_or_eval='train'):
        self.n_recordings = n_recordings
        self.max_recording_mins = max_recording_mins
        self.preproc_functions = preproc_functions
        self.train_or_eval = train_or_eval

    def load(self):
        log.info("Read file names")
        all_file_names, labels = get_all_sorted_file_names_and_labels(
            train_or_eval=self.train_or_eval)

        log.info("Read recording lengths...")
        if self.max_recording_mins is not None:
            lengths = np.load(
                '/home/schirrmr/code/auto-diagnosis/sorted-recording-lengths.npy')
            mask = lengths < self.max_recording_mins * 60
            cleaned_file_names = np.array(all_file_names)[mask]
            cleaned_labels = labels[mask]
        else:
            cleaned_file_names = np.array(all_file_names)
            cleaned_labels = labels
        X = []
        y = []
        n_files = len(cleaned_file_names[:self.n_recordings])
        for i_fname, fname in enumerate(cleaned_file_names[:self.n_recordings]):
            log.info("Load {:d} of {:d}".format(i_fname + 1,n_files))
            x = load_data(fname, preproc_functions=self.preproc_functions)
            assert x is not None
            X.append(x)
            y.append(cleaned_labels[i_fname])
        y = np.array(y)
        return X, y