#!/usr/bin/env python3.5
import numpy as np
import argparse
import logging

from windowing import data_splitter
from cleaning import data_cleaner
from utils import my_io


def median_amplitudes_over_all_channels(ffted, sampling_frequency, window_size_sec, limits):
    window_size = window_size_sec * sampling_frequency
    amplitudes = np.abs(ffted)

    freq_bin_size = sampling_frequency / window_size
    median_amplitudes = []
    for band_limits in limits:
        # amplitudes shape: windows x electrodes x frequencies
        band_amplitudes = amplitudes[:, :, int(band_limits[0] / freq_bin_size):int(band_limits[1] / freq_bin_size)]
        median_amplitude_band = np.median(band_amplitudes, axis=2)
        median_amplitude_windows = np.median(median_amplitude_band, axis=0)
        median_amplitude_ch = np.median(median_amplitude_windows)
        median_amplitudes.append(median_amplitude_ch)

    return median_amplitudes


def main(cmd_args):
    logging.basicConfig(level=cmd_args.logging,
                        format="%(asctime)s - [%(levelname)s] (%(module)s:%(lineno)d) %(message)s")

    data_set = cmd_args.input.split('/')[-2]

    limits = [(1, 4), (4, 8), (8, 12), (12, 30), (30, 90)]
    if cmd_args.finer:
        limits = [(1, 2.5), (2.5, 4), (4, 6), (6, 8), (8, 10), (10, 12), (12, 21), (21, 30), (30, 60), (60, 90)]

    cleaner = data_cleaner.DataCleaner()
    splitter = data_splitter.DataSplitter(window=cmd_args.window)

    features = []
    edf_files = my_io.read_all_file_names(cmd_args.input, extension='.edf')
    for edf_file in edf_files[cmd_args.start:cmd_args.end]:
        logging.info("Processing: {}".format(edf_file))

        # there is one 24h recording, which is just too large that also has more artifacts than actual data
        if my_io.get_recording_length(edf_file) > 14400:
            logging.warning("Recording is too long ({}s)!".format(my_io.get_recording_length(edf_file)))
            continue

        signals, sampling_frequency, n_samples, n_signals, signal_names, duration = my_io.get_data_with_mne(edf_file)
        if signals is None or sampling_frequency < 100:
            logging.warning("Recording has a weird sampling frequency ({}Hz).".format(sampling_frequency))
            continue

        signals, used_electrodes = cleaner.clean(signals, sampling_frequency, signal_names, duration)
        if signals is None:
            logging.warning("Recording is too short ({}s).".format(duration))
            continue

        epochs = splitter.split(signals, sampling_frequency, cmd_args.windowsize)
        epochs_ft = np.fft.rfft(epochs, axis=2)

        # TODO: compute more feautres here
        amplitude_features = median_amplitudes_over_all_channels(epochs_ft, sampling_frequency, cmd_args.windowsize,
                                                                 limits)
        features.append(amplitude_features)

    features = np.vstack(features)

    # TODO: store all relevant information in hdf5 header. e.g. window type, window length, all features, ...
    header_data = ['median delta amp', 'median theta amp', 'median alpha amp', 'median beta amp', 'median gamma amp']
    if cmd_args.finer:
        header_data = ['median lower delta amp', 'median upper delta amp', 'median lower theta amp',
                       'median upper theta amp', 'median lower alpha amp', 'median upper alpha amp',
                       'median lower beta amp', 'median upper beta amp', 'median lower gamma amp',
                       'median upper beta amp']

    my_io.write_hdf5(cmd_args.output, [data_set, cmd_args.window, str(cmd_args.windowsize), str(cmd_args.finer)],
                     features, header_data)


if __name__ == '__main__':
    windows = data_splitter.DataSplitter.get_supported_windows()

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('input', type=str, help='root dir of edf files')
    parser.add_argument('output', type=str, help='output directory')

    parser.add_argument('-w', '--window', type=str, default='boxcar', choices=windows, metavar='',
                        help='window function to improve ft. choose between: ' + ', '.join(windows))
    parser.add_argument('-ws', '--windowsize', type=int, help='length of the window in seconds', default=2, metavar='')
    parser.add_argument('-f', '--finer', action='store_true', help='splits all frequency bands into two')

    parser.add_argument('-s', '--start', type=int, default=None, metavar='', help='start index of edf files')
    parser.add_argument('-e', '--end', type=int, default=None, metavar='', help='stop index of edf files')

    parser.add_argument('-l', '--logging', type=int, default=20, metavar='', choices=[0, 10, 20, 30, 40, 50],
                        help='logging verbosity level: ' + ', '.join(str(s) for s in [0, 10, 20, 30, 40, 50]))

    cmd_arguments, unknown = parser.parse_known_args()
    main(cmd_arguments)



