import pandas as pd
import numpy as np
import logging
import scipy

import pyeeg
import pywt

from feature_generation import features_frequency
from feature_generation import features_time


class FeatureGenerator(object):
    """ Generates a compilation of fft, time, corr and dwt features """

# ______________________________________________________________________________________________________________________
    def introduce(self):
        msg = "\t\tHi, I am the feature generator! I am part of the preprocessing step. I compute a hopefully " \
              "meaningful representation of your data using the following set of features: {}."\
            .format(', '.join(self.get_info()))

        if self.perrec:
            logging.info(msg + " The features are computed per recording.\n")
        else:
            logging.info(msg + " The features are computed per window.\n")

# its not over all channels anymore!
# ______________________________________________________________________________________________________________________
    def compute_freq_feats(self, freq_feat_name, rec):
        """ Computes the feature values for a given recording. Computes the value the frequency bands as specified in
        band_limits. The values are mean over all time windows and channels
        :param freq_feat_name: the function name that should be called
        :param rec: the recording object holding the data and info
        :return: mean amplitudes in frequency bands over the different channels
        """
        func = getattr(features_frequency, freq_feat_name)
        # amplitudes shape: windows x electrodes x frequencies
        amplitudes = np.abs(rec.signals_ft)
        window_size = self.window_size_sec * rec.sampling_freq
        freq_bin_size = rec.sampling_freq / window_size

        mean_amplitudes = []
        for i in range(len(self.bands) - 1):
            lower, upper = self.bands[i], self.bands[i+1]
            # add the suggested band overlap of 50%
            if i != 0:
                lower -= 0.5 * (self.bands[i] - self.bands[i-1])
            if i != len(self.bands) - 2:
                upper += 0.5 * (self.bands[i+2] - self.bands[i+1])

            lower_bin, upper_bin = int(lower / freq_bin_size), int(upper / freq_bin_size)
            band_amplitudes = amplitudes[:, :, lower_bin:upper_bin]

            mean_amplitude_bands = func(band_amplitudes, axis=2)
            mean_amplitude_windows = np.mean(mean_amplitude_bands, axis=0)
            mean_amplitudes.extend(list(mean_amplitude_windows))

        return mean_amplitudes

# ______________________________________________________________________________________________________________________
    def rolling_to_windows(self, rolling_feature, window_size):
        """ This should be used to transform the results of the rolling operation of pandas to window values.
        beware! through ".diff" in pandas feature computation, a row of nan is inserted to the rolling feature.
        :param rolling_feature: feature computation achieved through pandas.rolling()
        :param window_size: number of samples in the window
        :return: rolling feature sliced at the window locations
        """
        overlap_size = int(self.overlap * window_size)
        windowed_feature = rolling_feature[window_size-1::window_size-overlap_size]
        return windowed_feature

# ______________________________________________________________________________________________________________________
    def bin_power(self, X, Band, Fs):
        """ taken from pyeeg lib and adapted since on cluster slicing failed through float indexing.
        :param X: 1d signal
        :param Band: frequency band
        :param Fs: sampling frequency
        :return: power and power ratio of frequency band
        """
        C = np.fft.fft(X)
        C = abs(C)
        Power = np.zeros(len(Band) - 1)
        for Freq_Index in range(0, len(Band) - 1):
            Freq = float(Band[Freq_Index])
            Next_Freq = float(Band[Freq_Index + 1])
            Power[Freq_Index] = sum(
                C[int(Freq / Fs * len(X)): int(Next_Freq / Fs * len(X))]
            )
        Power_Ratio = Power / sum(Power)
        return Power, Power_Ratio

# ______________________________________________________________________________________________________________________
    def spectral_entropy(self, X, Band, Fs, Power_Ratio=None):
        """ taken from pyeeg lib and adapted to use self.bin_power since it returned nan
        :param X:
        :param Band:
        :param Fs:
        :param Power_Ratio:
        :return:
        """
        if Power_Ratio is None:
            Power, Power_Ratio = self.bin_power(X, Band, Fs)

        # added to catch crashes
        if len(Power_Ratio) == 1:
            return 0

        Spectral_Entropy = 0
        for i in range(0, len(Power_Ratio) - 1):
            Spectral_Entropy += Power_Ratio[i] * np.log(Power_Ratio[i])
        Spectral_Entropy /= np.log(
            len(Power_Ratio)
        )  # to save time, minus one is omitted
        return -1 * Spectral_Entropy

# ______________________________________________________________________________________________________________________
#     def compute_corr_feats(self, rec):
#         auto_corr, cross_corr = [], []
#         correlation_coefficients = []
#
#         print(rec.signal_names)
#         # rec.signals has shape n_windows x n_channels x n_samples_in_window
#         for correlation_pair in self.correlation_pairs:
#             (elec1, elec2) = correlation_pair
#             id1, id2 = rec.signal_names.index(elec1), rec.signal_names.index(elec2)
#             elec1_values, elec2_values = rec.signals[:, id1, :], rec.signals[:, id2, :]
#             print(elec1_values.shape)
#
#             for window_id, window in enumerate(elec1_values):
#                 correlation_coefficients.extend(np.correlate(window, elec2_values[window_id]))
#             print(correlation_coefficients)
#
#         correlation_coefficients = np.vstack(correlation_coefficients)
#         print(correlation_coefficients.shape)
#         # return auto_corr, cross_corr

# ______________________________________________________________________________________________________________________
    def compute_pyeeg_feats(self, rec):
        # these values are taken from the tuh paper
        TAU, DE, Kmax = 4, 10, 5
        pwrs, pwrrs, pfds, hfds, mblts, cmplxts, ses, svds, fis, hrsts = [], [], [], [], [], [], [], [], [], []
        dfas, apes = [], []

        for window_id, window in enumerate(rec.signals):
            for window_electrode_id, window_electrode in enumerate(window):
                # taken from pyeeg code / paper
                electrode_diff = list(np.diff(window_electrode))
                M = pyeeg.embed_seq(window_electrode, TAU, DE)
                W = scipy.linalg.svd(M, compute_uv=False)
                W /= sum(W)

                power, power_ratio = self.bin_power(window_electrode, self.bands, rec.sampling_freq)
                pwrs.extend(list(power))
                # mean of power ratio is 1/(len(self.bands)-1)
                pwrrs.extend(list(power_ratio))

                pfd = pyeeg.pfd(window_electrode, electrode_diff)
                pfds.append(pfd)

                hfd = pyeeg.hfd(window_electrode, Kmax=Kmax)
                hfds.append(hfd)

                mobility, complexity = pyeeg.hjorth(window_electrode, electrode_diff)
                mblts.append(mobility)
                cmplxts.append(complexity)

                se = self.spectral_entropy(window_electrode, self.bands, rec.sampling_freq, power_ratio)
                ses.append(se)

                svd = pyeeg.svd_entropy(window_electrode, TAU, DE, W=W)
                svds.append(svd)

                fi = pyeeg.fisher_info(window_electrode, TAU, DE, W=W)
                fis.append(fi)

                # this crashes...
                # ape = pyeeg.ap_entropy(electrode, M=10, R=0.3*np.std(electrode))
                # apes.append(ape)

                # takes very very long to compute
                # hurst = pyeeg.hurst(electrode)
                # hrsts.append(hurst)

                # takes very very long to compute
                # dfa = pyeeg.dfa(electrode)
                # dfas.append(dfa)

        pwrs = np.asarray(pwrs).reshape(rec.signals.shape[0], rec.signals.shape[1], len(self.bands)-1)
        pwrs = np.mean(pwrs, axis=0)

        pwrrs = np.asarray(pwrrs).reshape(rec.signals.shape[0], rec.signals.shape[1], len(self.bands)-1)
        pwrrs = np.mean(pwrrs, axis=0)

        pfds = np.asarray(pfds).reshape(rec.signals.shape[0], rec.signals.shape[1])
        pfds = np.mean(pfds, axis=0)

        hfds = np.asarray(hfds).reshape(rec.signals.shape[0], rec.signals.shape[1])
        hfds = np.mean(hfds, axis=0)

        mblts = np.asarray(mblts).reshape(rec.signals.shape[0], rec.signals.shape[1])
        mblts = np.mean(mblts, axis=0)

        cmplxts = np.asarray(cmplxts).reshape(rec.signals.shape[0], rec.signals.shape[1])
        cmplxts = np.mean(cmplxts, axis=0)

        ses = np.asarray(ses).reshape(rec.signals.shape[0], rec.signals.shape[1])
        ses = np.mean(ses, axis=0)

        svds = np.asarray(svds).reshape(rec.signals.shape[0], rec.signals.shape[1])
        svds = np.mean(svds, axis=0)

        fis = np.asarray(fis).reshape(rec.signals.shape[0], rec.signals.shape[1])
        fis = np.mean(fis, axis=0)

        return list(pwrs.ravel()), list(pwrrs.ravel()), pfds, hfds, mblts, cmplxts, ses, svds, fis, apes, hrsts, dfas

# ______________________________________________________________________________________________________________________
    def compute_dwt_feats(self, rec_coeffs):
        # TODO: compute this per time window?
        """ compute features on sub-bands achieved through multi level dwt
        :param rec_coeffs: list of length n_levels with n_electrodes x n_coeffs entries
        :return: mean of absolute values of coeffs of each sub-band, average power of the coeffs in each sub-band
                 std of coeffs in each sub-band, ratio of absolute mean values of adjacent sub-bands
        """
        means, avg_powers, stds, ratios = [], [], [], []
        for e in rec_coeffs:
            means.append(np.mean(np.abs(e), axis=1))
            avg_powers.append(np.mean(e * e, axis=1))
            stds.append(np.std(e, axis=1))

        means = np.asarray(means)
        avg_powers = np.asarray(avg_powers)
        stds = np.asarray(stds)

        # for every electrode, take the ratio of means of adjacent sub-bands
        for elec in range(means.shape[1]):
            for lvl in range(1, means.shape[0]):
                ratios.append(means[lvl-1][elec] / means[lvl][elec])

        return means.ravel(), avg_powers.ravel(), stds.ravel(), np.asarray(ratios).ravel()

# ______________________________________________________________________________________________________________________
    def generate_features_perrec(self, rec):
        """ computes features. returns one feature vector per recording
        :param rec: rec.signals has shape n_windows x n_electrodes x n_samples_in_window
                    rec.signals_ft has shape n_windows x n_electrodes x n_samples_in_window/2 + 1
        :return: the feature vector of that recording
        """
        # the window size in seconds can be float. make sure that window_size is int. if not pandas rolling crashes
        window_size = int(self.window_size_sec * rec.sampling_freq)
        features = list()

        # adds 2 patient features
        # TODO: add medication, ...?
        if self.patient_feat_flag:
            features.append(rec.sex)  # does not seem to have any influence at all
            features.append(rec.age)  # seems to have high influence

########################################################################################################################
        # frequency features
        # computes n_features * n_bands * n_electrodes features
        # TODO: test
        if self.freq_feat_flag:
            for freq_feat_name in self.freq_feats:
                # func = getattr(features_frequency, freq_feat_name)
                feature_values = self.compute_freq_feats(freq_feat_name, rec)
                features.extend(feature_values)

########################################################################################################################
        # time features speeded up by computation on whole signals with a rolling window
        # TODO: ".diff() in computation moves results by 1 position", how to pick the correct values from the rolling?!
        if self.time_feat_flag:
            for time_feat in self.time_feats:
                func = getattr(features_time, time_feat)
                # TODO: don't cast the signals here, originally safe them as pandas DataFrame
                feature_values = func(pd.DataFrame(rec.signals_complete.T), window_size)
                feature_values = self.rolling_to_windows(feature_values, window_size)
                feature_values = np.mean(feature_values, axis=0)
                features.extend(feature_values)

########################################################################################################################
        # These values are taken from the pyeeg paper
        # pyeeg features. hurst and dfa have a very, very high computation time.
        # implementations are given on 1D time series only. therefore pass every channel of the recording to the module
        # computes 7*n_elec + 2*n_bands*n_elecs values
        #  TODO: test and speed up
        if self.pyeeg_feat_flag:
            pwrs, pwrrs, pfds, hfds, mblts, cmplxts, ses, svds, fis, apes, hrsts, dfas = self.compute_pyeeg_feats(rec)

            features.extend(pwrs)
            features.extend(pwrrs)
            features.extend(pfds)
            features.extend(hfds)
            features.extend(mblts)
            features.extend(cmplxts)
            features.extend(ses)
            features.extend(svds)
            features.extend(fis)

            # features.extend(apes)
            # features.extend(hrsts)
            # features.extend(dfas)

########################################################################################################################
        # correlation features
        # takes too long to compute on whole signals
        # correlate opposite electrodes?
        # TODO: add
        if self.corr_feat_flag:
            autocorrelations, crosscorrelations = [], []

            features.extend(autocorrelations)
            features.append(crosscorrelations)

########################################################################################################################
        # dwt features
        # computes 3*n_elecs*n_lvls + n_elecs*(n_lvls-1)
        # TODO: check sampling frequency and maybe resample to achieve equal sub-bands?
        # TODO: try the "morlet" wavelet -> cwt?
        # TODO: test
        if self.dwt_feat_flag:
            wavelet = pywt.Wavelet('db4')
            level = pywt.dwt_max_level(window_size, wavelet.dec_len)
            if level > 5:
                level = 5
            # TODO: handle this case
            elif level < 5:
                logging.warning("Recording {} is too short to perform a level-{} dwt!".format(rec.name, 5))

            # rec_coeffs is in shape: n_levels x n_electrodes x n_band_coeffs
            rec_coeffs = pywt.wavedec(rec.signals_complete, wavelet=wavelet, level=level)
            # don't use band d1?
            # rec_coeffs = rec_coeffs[1:]

            means, avg_powers, stds, ratios = self.compute_dwt_feats(rec_coeffs)
            features.extend(means)
            features.extend(avg_powers)
            features.extend(stds)
            features.extend(ratios)

        return features

# ______________________________________________________________________________________________________________________
    def generate_features_perwin(self, rec):
        """ this function starts feature computation on time windows instead of whole signals
        :param rec: the recording
        :return: one feature vector for every time window, a feature matrix with n_electrodes * n_windows x n_features
        """
        # features = []
        # for feature_f in self.features_to_compute:
        #     func = getattr(np, feature_f)
        #     median_in_window = func(rec.signals_ft, axis=2)
        #     median_over_channels = np.median(median_in_window, axis=1)
        #     feature_values = np.vstack(median_over_channels).T
        #     features.append(feature_values)
        #
        # features = np.vstack(features).T
        # return features
        return list()

# ______________________________________________________________________________________________________________________
    def generate_features(self, rec):
        """ Compute features either per recording or per time window """
        if not self.perrec:
            return self.generate_features_perwin(rec)

        return self.generate_features_perrec(rec)

# ______________________________________________________________________________________________________________________
    def set_reference(self, rec):
        # self.reference_recording = rec
        logging.info("\t\tSet {} as a reference.".format(rec.name))
        return

# ______________________________________________________________________________________________________________________
    def get_info(self):
        if self.feature_names is None:
            feat_names = []
            if self.patient_feat_flag:
                feat_names.extend(self.patient_feats)
            if self.freq_feat_flag:
                feat_names.extend(self.freq_feats)
            if self.time_feat_flag:
                feat_names.extend(self.time_feats)
            if self.pyeeg_feat_flag:
                feat_names.extend(self.pyeeg_feats)
            if self.corr_feat_flag:
                feat_names.extend(self.corr_feats)
            if self.dwt_feat_flag:
                feat_names.extend(self.dwt_feats)
            self.feature_names = feat_names
        return self.feature_names

# ______________________________________________________________________________________________________________________
    def get_feature_labels(self):
        """ create a list of feature labels that can be used to identify the feature values later on """
        if self.feature_labels is None:
            feature_labels = list()

            # patient features
            if self.patient_feat_flag:
                for patient_feat in self.patient_feats:
                    feature_labels.append('_'.join(['patient', patient_feat]))

            # electrode names
            if self.electrodes is not None:
                electrodes = self.electrodes
            else:
                electrodes = np.linspace(1, 21, 21)
                electrodes = ['ch'+str(electrode) for electrode in electrodes]

            # fft features
            if self.freq_feat_flag:
                for freq_feat in self.freq_feats:
                    for band_id, band in enumerate(self.bands[:-1]):
                        for electrode in electrodes:
                            label = '_'.join(['fft', freq_feat, str(band) + '-' + str(self.bands[band_id+1]) + 'Hz',
                                              str(electrode)])
                            feature_labels.append(label)

            # time features (from master project)
            if self.time_feat_flag:
                for time_feat in self.time_feats:
                    for electrode in electrodes:
                        label = '_'.join(['time', time_feat, str(electrode)])
                        feature_labels.append(label)

            # pyeeg features
            if self.pyeeg_feat_flag:
                for pyeeg_feat in self.pyeeg_feats:
                    if pyeeg_feat == "pwr" or pyeeg_feat == "pwrr":
                        for band_id, band in enumerate(self.bands[:-1]):
                            for electrode in electrodes:
                                label = '_'.join(['pyeeg', pyeeg_feat, str(band) + '-' + str(self.bands[band_id + 1])
                                                  + 'Hz', str(electrode)])
                                feature_labels.append(label)
                    else:
                        for electrode in electrodes:
                            label = '_'.join(['pyeeg', pyeeg_feat, str(electrode)])
                            feature_labels.append(label)

            # correlation features
            if self.corr_feat_flag:
                for corr_feat in self.corr_feats:
                    feature_labels.append(corr_feat)

            # dwt features
            if self.dwt_feat_flag:
                dwt_bands = ["a5", "d5", "d4", "d3", "d2", "d1"]
                for dwt_feat in self.dwt_feats:
                    if dwt_feat == 'ratio':
                        for electrode in electrodes:
                            for band_id in range(0, len(dwt_bands)-1):
                                feature_labels.append('_'.join(['dwt', dwt_feat, str(dwt_bands[band_id]) + '-' +
                                                                str(dwt_bands[band_id+1]), electrode]))
                    else:
                        for electrode in electrodes:
                            for band in dwt_bands:
                                feature_labels.append('_'.join(['dwt', dwt_feat, band, str(electrode)]))

            self.feature_labels = np.asarray(feature_labels)

        return self.feature_labels

# ______________________________________________________________________________________________________________________
    def __init__(self, domain, bands, window_size_sec=2, overlap=0, perrec=True, electrodes=None):
        # all the features that are implemented
        self.patient_feats = ["sex", "age"]
        self.freq_feats = sorted([feat_func for feat_func in dir(features_frequency) if not feat_func.startswith('_')])
        self.time_feats = sorted([feat_func for feat_func in dir(features_time) if not feat_func.startswith('_')])
        pyeeg_freq_feats = ["pwr", "pwrr"]
        pyeeg_time_feats = ["pfd", "hfd", "mblt", "cmplxt", "se", "svd", "fi"]  # , "ape", "hrst", "dfa"]
        self.pyeeg_feats = sorted(pyeeg_freq_feats + pyeeg_time_feats)
        self.corr_feats = []
        self.dwt_feats = ["mean", "avg-power", "std", "ratio"]

        # TODO: add toggle computation of electrode sub groups. does this really make sense? would be best if all
        # TODO: features are always computed
        # toggle computation of feature sub groups
        self.corr_feat_flag = False
        if 'all' in domain:
            self.patient_feat_flag, self.freq_feat_flag, self.time_feat_flag, self.pyeeg_feat_flag, self.dwt_feat_flag \
                = True, True, True, True, True
        else:
            self.patient_feat_flag = True if 'patient' in domain else False
            self.freq_feat_flag = True if 'fft' in domain else False
            self.time_feat_flag = True if 'time' in domain else False
            self.pyeeg_feat_flag = True if 'pyeeg' in domain else False
            self.dwt_feat_flag = True if 'dwt' in domain else False
            self.corr_feat_flag = True if 'corr' in domain else False

        # cmd args needed for computation
        self.window_size_sec = window_size_sec
        self.overlap = overlap
        self.perrec = perrec
        self.bands = [int(digit) for digit in bands.split(',')]

        # information that is needed by other objects / written to file
        self.electrodes = electrodes
        self.feature_labels = None
        self.feature_names = None

        self.introduce()
