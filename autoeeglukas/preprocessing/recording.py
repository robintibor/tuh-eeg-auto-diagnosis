class Recording(object):
    """ This is a container class for all the relevant data of a EEG recording
    """

# ______________________________________________________________________________________________________________________
    def __init__(self, data_set, name, raw_edf, sampling_freq, n_samples, n_signals, signal_names, duration,
                 signals=None, signals_complete=None, signals_ft=None, sex=None, age=None):
        self.data_set = data_set
        self.name = name
        self.raw_edf = raw_edf
        self.sampling_freq = sampling_freq
        self.n_samples = n_samples
        self.n_signals = n_signals
        self.signal_names = signal_names
        self.duration = duration
        self.signals = signals
        self.signal_ft = signals_ft
        self.signals_complete = signals_complete
        self.sex = sex
        self.age = age
