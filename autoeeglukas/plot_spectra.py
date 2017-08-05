#!/usr/bin/env python3.5
import numpy as np
import argparse
import logging

from windowing import data_splitter
from visualization import plotting
from cleaning import data_cleaner
from utils import my_io


def main(cmd_args):
    logging.basicConfig(level=cmd_args.logging,
                        format="%(asctime)s - [%(levelname)s] (%(module)s:%(lineno)d) %(message)s")

    cleaner = data_cleaner.DataCleaner()
    splitter = data_splitter.DataSplitter(window=cmd_args.window)

    file_paths = my_io.read_all_file_names(cmd_args.input, extension=".edf")
    for file_path in file_paths[cmd_args.start:cmd_args.end]:
        logging.info("Processing: {}".format(file_path))

        # there is one 24h recording, which is just too large and has more artifacts than actual data
        if my_io.get_recording_length(file_path) > 14400:
            logging.warning("Recording is too long ({}s)!".format(my_io.get_recording_length(file_path)))
            continue

        data, sampling_frequency, n_samples, n_signals, signal_names, duration = my_io.get_data_with_mne(file_path)
        if data is None or sampling_frequency < 100:
            logging.warning("Recording has a weird sampling frequency ({}Hz).".format(sampling_frequency))
            continue

        used_data, used_electrodes = cleaner.clean(data, sampling_frequency, signal_names, duration)
        if used_data is None:
            logging.warning("Recording is too short ({}s).".format(duration))
            continue

        window_size = int(cmd_args.windowsize*sampling_frequency)
        data_windowed = splitter.split(used_data, sampling_frequency, cmd_args.windowsize)

        data_ft = np.fft.rfft(data_windowed, axis=2)

        spectrum = np.abs(data_ft)
        spectrum_median = np.median(spectrum, axis=0)

        frequency_vector = np.fft.rfftfreq(n=window_size, d=1./sampling_frequency)

        edf_file_name_tokens = file_path.split('/')[-4:]
        edf_file_name = '/'.join(edf_file_name_tokens)

        plotting.plot_spectrum(spectrum_median, edf_file_name, duration, sampling_frequency, window_size,
                               used_electrodes, frequency_vector, cmd_args.output)

        # free memory
        del data, used_data, data_ft, spectrum, spectrum_median


if __name__ == '__main__':
    windows = data_splitter.DataSplitter.get_supported_windows()

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('input', type=str, help='root dir of edf files')
    parser.add_argument('output', type=str, help='output directory')

    parser.add_argument('-w', '--window', type=str, default='boxcar', choices=windows, metavar='',
                        help='window function to improve ft: ' + ', '.join(windows))
    parser.add_argument('-ws', '--windowsize', type=int, help='length of the window in seconds', default=2, metavar='')

    parser.add_argument('-s', '--start', type=int, default=None, metavar='', help='start index of edf files')
    parser.add_argument('-e', '--end', type=int, default=None, metavar='', help='stop index of edf files')

    parser.add_argument('-l', '--logging', type=int, default=20, metavar='', choices=[0, 10, 20, 30, 40, 50],
                        help='logging verbosity level: ' + ', '.join(str(s) for s in [0, 10, 20, 30, 40, 50]))

    cmd_arguments, unknown = parser.parse_known_args()
    main(cmd_arguments)
