import os
os.sys.path.insert(0, '/home/schirrmr/code/auto-diagnosis/')
import logging
import sys
import resampy
import numpy as np
from braindecode.datautil.signal_target import SignalAndTarget
from braindecode.datautil.iterators import CropsFromTrialsIterator
from autodiag.dataset import DiagnosisSet
import scipy.signal

log = logging.getLogger(__name__)


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(levelname)s : %(message)s',
                     level=logging.DEBUG, stream=sys.stdout)
    max_abs_val = 800
    sec_to_cut = 60
    duration_recording_mins = 20
    n_recordings = 3000#3000
    sampling_freq = 100
    divisor = 10
    max_recording_mins = 35
    preproc_functions = []
    preproc_functions.append(
        lambda data, fs: (data[:, int(sec_to_cut * fs):-int(
            sec_to_cut * fs)], fs))
    preproc_functions.append(
        lambda data, fs: (data[:, :int(duration_recording_mins * 60 * fs)], fs))

    preproc_functions.append(lambda data, fs:
                             (np.clip(data, -max_abs_val, max_abs_val), fs))

    preproc_functions.append(lambda data, fs: (resampy.resample(data, fs,
                                                                sampling_freq,
                                                                axis=1,
                                                                filter='kaiser_fast'),
                                               sampling_freq))

    preproc_functions.append(lambda data, fs: (data / divisor, fs))
    dataset = DiagnosisSet(n_recordings=n_recordings,
                           max_recording_mins=max_recording_mins,
                           preproc_functions=preproc_functions,
                           train_or_eval='train')

    X, y = dataset.load()
    dataset = SignalAndTarget(X, y)

    batch_size = 64
    input_time_length = 1200
    n_preds_per_input = 600
    iterator = CropsFromTrialsIterator(batch_size=batch_size,
                                       input_time_length=input_time_length,
                                       n_preds_per_input=n_preds_per_input)

    w = scipy.signal.windows.blackmanharris(input_time_length)
    w = w / np.linalg.norm(w)
    w = w.astype(np.float32)
    for setname, y_class in (('normal', 0), ('abnormal', 1)):
        this_X = [x for x, y in zip(dataset.X, dataset.y) if y == y_class]

        this_y = dataset.y[dataset.y == y_class]

        assert np.all(this_y == y_class)
        this_set = SignalAndTarget(this_X, this_y)
        log.info("Create batches...")
        batches_arr = [b[0] for b in iterator.get_batches(this_set, shuffle=False)]
        log.info("Create batches array...")
        batches_arr = np.concatenate(batches_arr, axis=0)
        batches_arr = batches_arr * w[None,None,:, None]
        log.info("Compute FFT...")
        # fensterfunktion
        bps = np.square(np.abs(np.fft.rfft(batches_arr, axis=2)).astype(np.float32))
        del batches_arr
        log.info("Compute median...")
        median_bp = np.median(bps, axis=(0, 3))
        log.info("Compute mean...")
        mean_bp = np.mean(bps, axis=(0, 3))
        del bps
        median_filename = "median-{:s}-bps-windowed.npy".format(setname)
        log.info("Saving to {:s}...".format(median_filename))
        np.save(median_filename, median_bp)
        mean_filename = "mean-{:s}-bps-windowed.npy".format(setname)
        log.info("Saving to {:s}...".format(mean_filename))
        np.save(mean_filename, mean_bp)
