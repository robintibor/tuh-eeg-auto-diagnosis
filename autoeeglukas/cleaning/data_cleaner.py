import numpy as np
import logging
import mne

from utils import my_io

ELECTRODES = np.asarray(sorted([
    ' FP1', ' FP2',
    ' F3', ' F4',
    # ' C3-', ' C4-', ' C3 ', ' C4 ',  # to not find C3P / DC
    ' C3', ' C4',
    ' P3', ' P4',
    ' O1', ' O2',
    ' F7', ' F8',
    ' T3', ' T4',
    ' T5', ' T6',
    ' A1', ' A2',  # not present in all recordings
    # ' FPZ',  # not standard
    ' FZ',
    ' CZ',
    ' PZ',
    # ' OZ',  # not standard
    # ' PG1', ' PG2',  # not standard
]))


class DataCleaner(object):
    """
    """

    @staticmethod
# ______________________________________________________________________________________________________________________
    def get_supported_electrodes():
        return ELECTRODES

# ______________________________________________________________________________________________________________________
    def introduce(self):
        logging.info("\t\tHi, I am the data cleaner. I am part of the preprocessing step. I use a time threshold of "
                     "{}s to find too short recordings, a start and end time shift of {}% and {}% to remove artifacts, "
                     "will filter out the power line frequency of {} Hz and fit the frequencies to [{}, {}]. I will "
                     "take the electrodes according to the 10-20-system: {}."
                     .format(self.time_threshold, self.start_time_shift*100, self.end_time_shift*100,
                             self.power_line_freq, self.low_cut, self.high_cut, ', '.join(self.electrodes_to_use)))

# ______________________________________________________________________________________________________________________
    def take_subsets_of_elecs(self):
        """ takes a subset of electrodes as specified by cmd argument"""
        if 'all' in self.elecs:
            return
        else:
            # transform to upper case in case the user accidentally inputs lowercase
            self.elecs = [elec.upper() for elec in self.elecs]
            new_electrodes_to_use_ids = []
            for elec in self.elecs:
                for electrode_id, electrode in enumerate(self.electrodes_to_use):
                    if elec in electrode:
                        new_electrodes_to_use_ids.append(electrode_id)

            new_electrodes_to_use_ids = np.asarray(sorted(new_electrodes_to_use_ids))
            self.electrodes_to_use = self.electrodes_to_use[new_electrodes_to_use_ids]

# ______________________________________________________________________________________________________________________
    def bandpass_time_domain(self, rec):
        """ filters the signal to frequency range self.low_cut - self.high_cut """
        rec.signals = mne.filter.filter_data(rec.signals, rec.sampling_freq, self.low_cut, self.high_cut,
                                             verbose='error')
        return rec

# ______________________________________________________________________________________________________________________
    def filter_power_line_frequency(self, rec):
        """ Remove the power line frequency from the recordings """
        rec.signals = mne.filter.notch_filter(rec.signals, rec.sampling_freq, np.arange(self.power_line_freq,
                                                                                        rec.sampling_freq/2,
                                                                                        self.power_line_freq),
                                              verbose='error')
        return rec

# ______________________________________________________________________________________________________________________
    def remove_start_end_artifacts(self, rec):
        """ Removes self.start_time_shift percent of the recording from the beginning and self.end_time_shift from the
        end, since these parts often showed artifacts. """
        # use start and end time shift as percentages of the total signal
        start_time_shift = int(self.start_time_shift * rec.duration)
        end_time_shift = int(self.end_time_shift * rec.duration)

        rec.signals = rec.signals[:, start_time_shift * rec.sampling_freq:-end_time_shift * rec.sampling_freq]
        rec.duration = rec.duration - (start_time_shift + end_time_shift)

        return rec

# ______________________________________________________________________________________________________________________
    def crop_data(self, rec):
        """ Only pick those specific electrodes from the recording """
        used_data = rec.signals[rec.signal_names]
        # TODO: maybe not do this? trainer needs a dataframe anyway. does he?
        # transform pandas data frame to numpy array
        rec.signals = used_data.as_matrix().T
        return rec

# ______________________________________________________________________________________________________________________
    def has_10_20_electrodes(self, rec):
        """ Check if the recording has all the electrodes that should be processed """
        used_electrodes = []
        for electrode_to_use in self.electrodes_to_use:
            for signal_name in rec.signal_names:
                if electrode_to_use in signal_name:
                    if electrode_to_use == ' C3' or electrode_to_use == ' C4':
                        if 'DC' in signal_name or 'C3P' in signal_name or 'C4P' in signal_name:
                            # logging.info("Non-10-20-system electrode was almost selected!")
                            continue

                    # replace the name given in the file with an appreviation, e.g. o1, f3...
                    used_electrodes.append(electrode_to_use.replace('-', '').strip())

        rec.signal_names = used_electrodes
        # -2 because of the two variants of C3 and C4
        return len(used_electrodes) == len(self.electrodes_to_use)

# ______________________________________________________________________________________________________________________
    def exceeds_time_threshold(self, rec):
        return rec.duration > self.time_threshold

# ______________________________________________________________________________________________________________________
    def cut_one_minute(self, rec):
        """
        :param rec:
        :return:
        """
        midpoint = int(rec.signals.shape[1] / 2)
        thirty_secs = int(30 * rec.sampling_freq)
        rec.signals = rec.signals[:, midpoint - thirty_secs:midpoint + thirty_secs]
        rec.duration = rec.signals.shape[1] / rec.sampling_freq
        # this cutting does not make every recording have the same number of samples. they differ due to varying
        # sampling frequencies
        return rec

# ______________________________________________________________________________________________________________________
    def volts_to_microvolts(self, rec):
        rec.signals *= 1000000
        return rec

# ______________________________________________________________________________________________________________________
    def clean(self, rec):
        # take a subset of electrodes
        self.take_subsets_of_elecs()

        # looks like its not necessary to return the object. good style?
        rec = self.crop_data(rec)
        rec = self.remove_start_end_artifacts(rec)
        rec = self.filter_power_line_frequency(rec)
        # TODO: this seems to "recenter" the data! find out why and how
        rec = self.bandpass_time_domain(rec)

        # drastically reduce amount of data by grabbing 1 minute of recording from the middle
        # rec = self.cut_one_minute(rec)

        # transform signal amplitudes from volts to microvolts
        rec = self.volts_to_microvolts(rec)
        return rec

# ______________________________________________________________________________________________________________________
    def rec_ok(self, rec):
        """ check if recording is suitable without actually loading the data
        :param rec: the recoring object holding all data from an edf file
        :return: true if recording can be processed, false otherwise
        """
        # some files cannot be opened or have a weird sampling frequency that cannot be processed
        if rec.raw_edf is None:
            if rec.sampling_freq is None:
                logging.warning("\t\tRecording {} could not be opened.".format(rec.name))

            elif rec.sampling_freq < 25:
                logging.warning("\t\tRecording {} has a weird sampling frequency ({:.2f}Hz). Skipping.".
                                format(rec.name, rec.sampling_freq))
            return False

        # short_name = re.findall('\d*/\d*/s\d*_\d*_\d*_\d*/a_\d*.edf', rec.name)[0]
        # there is one 24h recording, which is just too large that also has more artifacts than actual data
        # old 14400
        if my_io.get_recording_length(rec.name) > 20000:
            logging.warning("\t\tRecording {} is too long ({}s)! Skipping.".format(rec.name, rec.duration))
            return False

        # some recordings are too short
        if not self.exceeds_time_threshold(rec):
            logging.warning("\t\tRecording {} is too short ({}s). Skipping.".format(rec.name, rec.duration))
            return False

        # if a recording does not have all the electrodes we are looking for
        if not self.has_10_20_electrodes(rec):
            logging.warning("\t\tRecording {} does not have all electrodes ({}). Skipping.".format(rec.name,
                                                                                                   rec.signal_names))
            return False
        return True

# ______________________________________________________________________________________________________________________
    def get_electrodes(self):
        return [electrode.strip() for electrode in self.electrodes_to_use]

# ______________________________________________________________________________________________________________________
    def __init__(self, time_threshold=100, start_time_shift=0.05, end_time_shift=0.05, power_line_frequency=60,
                 low_cut=.2, high_cut=100, elecs=['all']):

        self.time_threshold = time_threshold
        self.start_time_shift = start_time_shift
        self.end_time_shift = end_time_shift
        self.power_line_freq = power_line_frequency
        self.low_cut = low_cut
        self.high_cut = high_cut
        self.electrodes_to_use = ELECTRODES
        self.elecs = elecs

        self.introduce()
