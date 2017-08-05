from functools import partial
import multiprocessing as mp
import numpy as np
import logging
import os

from feature_generation import feature_generator
from preprocessing import recording
from windowing import data_splitter
from visualization import plotting
from cleaning import data_cleaner

from utils import my_io


class Preprocessor(object):
    """ The preprocessor takes raw input data and returns features. Thereby it cleans and splits the data. The
    processing is parallelized, s.t. reading of data, cleaning, splitting, fourier transform, visualization and feature
    computation are faster.
    """

# ______________________________________________________________________________________________________________________
    def introduce(self):
        logging.info("\tHi, I am the preprocessor. I am part of the pipeline and responsible for cleaning, splitting "
                     "and feature generation of your data.")

# ______________________________________________________________________________________________________________________
    def process(self, rec):
        """ each file is first read, then cleaned, the split with the specified window function and transformed with
        fourier transformation
        """
        rec.signals = my_io.load_data_with_mne(rec)
        rec = self.cleaner.clean(rec)

        # are standard signals needed somewhere? or can they be overwritten?
        rec.signals_complete = rec.signals
        windows = self.splitter.split(rec)
        rec.signals = windows
        rec.signals_ft = np.fft.rfft(windows, axis=2)

        rec.signals_ft = np.abs(rec.signals_ft)
        return rec

# ______________________________________________________________________________________________________________________
    def process_with_visualization(self, rec):
        """ if cmd_args.visualize is set, the process includes visualization of the raw signal, the cleaned signal and
        the inverse transformed fourier transform of the signal as well as their spectra to show the effects of cleaning
        rules and window function
        :param rec: the recording
        :return: the cleaned and split and transformed recording as well as the intermediate results for visulization
        """
        rec.signals = my_io.load_data_with_mne(rec)

        signal_processing_steps = list()
        # for visualization of the raw signal
        signal_processing_steps.append(rec.signals["O1"])

        rec = self.cleaner.clean(rec)
        # for visualization of the cleaned signal
        signal_processing_steps.append(rec.signals[rec.signal_names.index("O1")])

        # are standard signals needed somewhere? or can they be overwritten/deleted?
        # needed if pandas.rolling should be used
        rec.signals_complete = rec.signals
        windows = self.splitter.split(rec)
        rec.signals = windows
        rec.signals_ft = np.fft.rfft(windows, axis=2)

        # for visualization of the spectrum of the cleaned signal
        signal_processing_steps.append(rec.signals_ft[0][rec.signal_names.index("O1")])

        rec.signals_ft = np.abs(rec.signals_ft)
        return rec, signal_processing_steps

# ______________________________________________________________________________________________________________________
    def clean_split_transform_visualize_featurize(self, rec, cmd_args):
        """ the function that is executed for every recording in the in_q
        :param rec: the recording
        :param cmd_args: needed to check if visualization is set and if to know where to save the figures to
        :return: the features of that recording and the number of time windows (to maybe combine window features after
        prediction)
        """
        logging.info("\t\tProcessing {}".format(rec.name))

        if 'all' in cmd_args.visualize or 'pre' in cmd_args.visualize:
            rec, signal_processing_steps = self.process_with_visualization(rec)
            spectrum_median = np.median(rec.signals_ft, axis=0)

            frequency_vector = np.fft.rfftfreq(n=int(cmd_args.windowsize * rec.sampling_freq), d=1. / rec.sampling_freq)
            # TODO: add class to file name?
            plotting.plot_spectrum(spectrum_median, rec, cmd_args, frequency_vector)
            signal_processing_steps.append(spectrum_median[rec.signal_names.index("O1")])
            plotting.plot_rec_analysis(signal_processing_steps, rec.sampling_freq, cmd_args)
        else:
            rec = self.process(rec)

        features = self.feature_generator.generate_features(rec)

        # return features and the number of windows in which the recording was split
        return features, len(rec.signals)

# ______________________________________________________________________________________________________________________
    def worker(self, in_q, out_q, func, args_=None):
        """ The worker will take one item of the in_q and will work on the element as specified in
        clean_split_transform_visualize_featurize. It then puts the results back to the out_q
        :param in_q: the queue that holds all the recordings that should be processed
        :param out_q: the queue that holds the results of all processed recordings
        :param func: the function that is run for to process every recording
        :param args_: additional arguments of the function
        """
        while not in_q.empty():
            item, ident = in_q.get(timeout=1)
            try:
                if args_:
                    out_q.put((func(item, **args_), ident))
                else:
                    out_q.put((func(item), ident))

            except Exception as ex:
                logging.error("\tProcessing failed with {}.".format(str(ex)))

# ______________________________________________________________________________________________________________________
    def random_subset(self, edf_files, subset):
        """ if cmd_args.subset is set, don't read all the recordings but take a random subset of the specified size
        :param edf_files: list of all recordings on the hdd
        :param subset: the size of the subset
        :return: a list of randomly selected files of the specified size
        """
        ids = np.arange(len(edf_files))
        np.random.shuffle(ids)
        subset_ids = ids[:subset]
        return [edf_files[i] for i in sorted(subset_ids)]

# ______________________________________________________________________________________________________________________
    def add_recording_to_queue(self, in_dir, edf_files, in_q):
        """ for ech recording that was read,
        :param in_dir: the input directory which is used to extract the class of the recording
        :param edf_files: list of recording paths
        :param in_q: the queue that holds the recordings for processing
        :return: the queue that holds the recordings for parallel processing, the number of recordings that are suitable
        for processing, e.g. that could be opened and read, has an appropriate length, etc
        """
        edf_count, edf_not_ok = 0, 0
        for edf_ident, edf_file in enumerate(edf_files):
            # this is due to the header fixing which didn't work as smoothly
            try:
                rec = recording.Recording(in_dir, edf_file, *my_io.get_info_with_mne(edf_file))
            except ValueError:
                logging.warning("Can't get info from file {}!".format(rec.name))
                edf_not_ok += 1
                self.stats.n_corrupted_recordings += 1
                continue

            if self.cleaner.rec_ok(rec):
                # the very first recording of the first class will be set as reference electrode for cross correlation
                # this recording can hence not be used for training itself
                # if self.feature_generator.reference_recording is None:
                #     rec, dummy = self.process(rec, cmd_args)
                #     self.feature_generator.set_reference(rec)
                #     edf_not_ok += 1
                #     continue

                # this is due to the header fixing which didn't work as smoothly
                try:
                    rec.sex, rec.age = my_io.get_patient_info(rec.name)
                except ValueError:
                    logging.warning("Can't get sex and age from file {}!".format(rec.name))
                    edf_not_ok += 1
                    self.stats.n_corrupted_recordings += 1
                    continue

                # 0 is male, 1 is female
                self.stats.ages_m.append(rec.age) if rec.sex == 0 else self.stats.ages_f.append(rec.age)
                self.stats.rec_lengths.append(rec.duration)
                self.recording_names[in_dir].append('/'.join(rec.name.split('/')[-3:]))

                in_q.put((rec, edf_ident - edf_not_ok))
                edf_count += 1
            else:
                edf_not_ok += 1
                self.stats.n_corrupted_recordings += 1

        if edf_count == 0:
            logging.fatal("\t No file is suitable for further processing.")
            exit()
        logging.info("\t{} of {} files are suitable for further processing.".format(edf_count, len(edf_files)))

        return in_q, edf_count

# ______________________________________________________________________________________________________________________
    def read_files(self, in_dir, subset):
        """ first, read all files from the given directory, then if a subset is desired, create a random subset. return
        a list of file names
        """
        edf_files = my_io.read_all_file_names(in_dir, extension='.edf')

        if subset is not None:
            edf_files = self.random_subset(edf_files, subset)
        logging.info("\tFound {} files. Checking.".format(len(edf_files)))

        return edf_files

# ______________________________________________________________________________________________________________________
    def spawn_start_join_processes(self, cmd_args, in_q, out_q):
        partial_process = partial(self.clean_split_transform_visualize_featurize, cmd_args=cmd_args)
        processes = [mp.Process(target=self.worker, args=(in_q, out_q, partial_process))
                     for i in range(cmd_args.n_proc)]

        for process in processes:
            process.start()

        for process in processes:
            process.join()

# ______________________________________________________________________________________________________________________
    def catch_results(self, in_dir_id, in_dir, out_q, edf_count):
        """ catch the results returned by the individual parallel processes
        :param in_dir_id: the id of the input directory
        :param in_dir: the input directory path
        :param out_q: the queue holding the results
        :param edf_count: the number of edfs processed of that class. this is needed to create the feature matrix of
        the correct size and to assign each feature vector to the correct position in the matrix (the order in which
        they were read from the hdd (sorted by date))
        :return: the feature matrix
        """
        self.window_counts.append([])
        features = edf_count * [None]
        while not out_q.empty():
            (values, n_windows), ident = out_q.get()
            features[ident] = values
            self.window_counts[in_dir_id].append(n_windows)

        if len(features) == 0:
            logging.fatal("\t\tNo features were computed for {}.".format(in_dir))
            exit()

        return features

# ______________________________________________________________________________________________________________________
    def check_path(self, in_dir):
        # TODO: move this to my_io?
        """ check if the given path exists
        """
        if not os.path.isdir(in_dir) or in_dir[-1] != '/':
            logging.fatal("Input directory {} does not exist. Did you forget to set '--no-pre'? Or maybe it is just "
                          "a missing '/'.".format(in_dir))
            exit()
        logging.info("\tGoing through recordings in {}:".format(in_dir))

# ______________________________________________________________________________________________________________________
    def init_processing_units(self, cmd_args):
        self.cleaner = data_cleaner.DataCleaner(elecs=cmd_args.elecs)
        self.splitter = data_splitter.DataSplitter(window=cmd_args.window, window_size_sec=cmd_args.windowsize,
                                                   overlap=cmd_args.overlap)
        self.feature_generator = feature_generator.FeatureGenerator(domain=cmd_args.domain, bands=cmd_args.bands,
                                                                    window_size_sec=cmd_args.windowsize,
                                                                    overlap=cmd_args.overlap, perrec=cmd_args.perrec,
                                                                    electrodes=self.cleaner.get_electrodes())

# ______________________________________________________________________________________________________________________
    def preprocess(self, cmd_args):
        """ Checks if all the EEG recordings of the input directories can be processed. For every processable recording
        an entry in the multiprocessing queue is inserted. Results (features) are written to .hdf5 files per input dir.
        :param cmd_args:
        :return:
        """
        self.init_processing_units(cmd_args)

        # create output directory
        my_io.check_out(cmd_args.output, cmd_args.input)
        my_io.write_feature_labels(cmd_args.output, self.feature_generator.get_feature_labels())

        # set up multiprocessing
        manager = mp.Manager()
        in_q, out_q = manager.Queue(), manager.Queue()

        # stores for every class the hdf5 file name where features are stored
        feature_files = []
        for in_dir_id, in_dir in enumerate(cmd_args.input):
            self.check_path(in_dir)
            self.stats.n_classes += 1
            self.recording_names[in_dir] = list()

            edf_files = self.read_files(in_dir, cmd_args.subset)
            self.stats.n_recordings += len(edf_files)

            in_q, edf_count = self.add_recording_to_queue(in_dir, edf_files, in_q)

            self.spawn_start_join_processes(cmd_args, in_q, out_q)

            features = self.catch_results(in_dir_id, in_dir, out_q, edf_count)
            features = np.vstack(features)

            file_name = my_io.write_hdf5(features, in_dir, cmd_args)
            feature_files.append(file_name)

        # TODO: add this
        my_io.write_recording_names(cmd_args.output, cmd_args.input, self.recording_names)

        return feature_files, self.window_counts, self.stats

# ______________________________________________________________________________________________________________________
    def __init__(self, stats):
        self.cleaner = None
        self.splitter = None
        self.feature_generator = None

        self.stats = stats

        self.window_counts = []
        self.recording_names = {}

        self.introduce()
