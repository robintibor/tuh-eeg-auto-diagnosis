from scipy import signal
import numpy as np
import logging

WINDOWS = [
    'barthann',
    'bartlett',
    'blackman',
    'blackmanharris',
    'bohman',
    'boxcar',
    'cosine',
    'flattop',
    'hamming',
    'hann',
    'nuttall',
    'parzen',
    'triang'
]


class DataSplitter(object):
    """
    """

    @staticmethod
# ______________________________________________________________________________________________________________________
    def get_supported_windows():
        return WINDOWS

# ______________________________________________________________________________________________________________________
    def introduce(self):
        logging.info("\t\tHi, I am the data splitter. I am part of the preprocessing step. I split your data into {}s "
                     "chunks with a {} window and I use an overlap of {}%."
                     .format(self.window_size_sec, self.window, int(self.overlap * 100)))

# ______________________________________________________________________________________________________________________
    def windows_weighted(self, windows, window_size):
        """ weights the splitted signal by the specified window function
        :param windows: the signals splitted into time windows
        :param window_size: the number of samples in the window
        :return: the windows weighted by the specified window function
        """
        method_to_call = getattr(signal, self.window)
        window = method_to_call(window_size)

        return windows * window

# ______________________________________________________________________________________________________________________
    def split(self, rec):
        """ written by robin schirrmeister, adapted by lukas gemein
        :param rec: the recording object holding the signals and all information needed
        :return: the signals split into time windows of the specified size
        """
        window_size = int(rec.sampling_freq * self.window_size_sec)
        overlap_size = int(self.overlap * window_size)
        stride = window_size - overlap_size

        if stride == 0:
            logging.error("Time windows cannot have an overlap of 100%.")

        # written by robin tibor schirrmeister
        signal_crops = []
        for i_start in range(0, rec.signals.shape[-1] - window_size + 1, stride):
            signal_crops.append(np.take(rec.signals, range(i_start, i_start + window_size), axis=-1, ))

        return self.windows_weighted(np.array(signal_crops), window_size)

# ______________________________________________________________________________________________________________________
    def __init__(self, overlap=0, window='boxcar', window_size_sec=2):
        self.overlap = overlap/100
        self.window = window
        self.window_size_sec = window_size_sec

        self.introduce()
