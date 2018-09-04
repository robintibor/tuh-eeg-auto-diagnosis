import pandas as pd
import numpy as np
import pygsheets
import logging
import h5py
import glob
import mne
import os
import re


# ______________________________________________________________________________________________________________________
def natural_key(file_name):
    """ provides a human-like sorting key of a string """
    key = [int(token) if token.isdigit() else None for token in re.split(r'(\d+)', file_name)]
    return key



def session_key(file_name):
    """ sort the file name by session """
    return re.findall(r'(s\d*)_', file_name)

# ______________________________________________________________________________________________________________________
def time_key(file_name):
    """ provides a time-based sorting key """
    splits = file_name.split('/')
    [date] = re.findall(r'(\d{4}_\d{2}_\d{2})', splits[-2])
    date_id = [int(token) for token in date.split('_')]
    recording_id = natural_key(splits[-1])
    session_id = session_key(splits[-2])
    
    return date_id + session_id + recording_id


# ______________________________________________________________________________________________________________________
def read_all_file_names(path, extension, key="time"):
    """ read all files with specified extension from given path
    :param path: parent directory holding the files directly or in subdirectories
    :param extension: the type of the file, e.g. '.txt' or '.edf'
    :param key: the sorting of the files. natural e.g. 1, 2, 12, 21 (machine 1, 12, 2, 21) or by time since this is
    important for cv. time is specified in the edf file names
    """
    file_paths = glob.glob(path + '**/*' + extension, recursive=True)

    if key == 'time':
        return sorted(file_paths, key=time_key)
        
    elif key == 'natural':
        return sorted(file_paths, key=natural_key)


# ______________________________________________________________________________________________________________________
def fix_header(file_path):
    """ this was used to try to fix the corrupted header of recordings. not needed anymore since they were officially
    repaired by tuh
    """
    logging.warning("Couldn't open edf {}. Trying to fix the header ...".format(file_path))
    f = open(file_path, 'rb')
    content = f.read()
    f.close()
    
    header = content[:256]
    # print(header)

    # version = header[:8].decode('ascii')
    # patient_id = header[8:88].decode('ascii')
    # [age] = re.findall("Age:(\d+)", patient_id)
    # [sex] = re.findall("\s\w\s", patient_id)

    recording_id = header[88:168].decode('ascii')
    # startdate = header[168:176]
    # starttime = header[176:184]
    # n_bytes_in_header = header[184:192].decode('ascii')
    # reserved = header[192:236].decode('ascii')
    # THIS IS MESSED UP IN THE HEADER DESCRIPTION
    # duration = header[236:244].decode('ascii')
    # n_data_records = header[244:252].decode('ascii')
    # n_signals = header[252:].decode('ascii')
    
    date = recording_id[10:21]
    day, month, year = date.split('-')
    if month == 'JAN':
        month = '01'

    elif month == 'FEB':
        month = '02'

    elif month == 'MAR':
        month = '03'

    elif month == 'APR':
        month = '04'

    elif month == 'MAY':
        month = '05'

    elif month == 'JUN':
        month = '06'

    elif month == 'JUL':
        month = '07'

    elif month == 'AUG':
        month = '08'

    elif month == 'SEP':
        month = '09'

    elif month == 'OCT':
        month = '10'

    elif month == 'NOV':
        month = '11'

    elif month == 'DEC':
        month = '12'

    year = year[-2:]
    date = '.'.join([day, month, year])
    
    fake_time = '00.00.00'
    
    # n_bytes = int(n_bytes_in_header) - 256
    # n_signals = int(n_bytes / 256)
    # n_signals = str(n_signals) + '    '
    # n_signals = n_signals[:4]
    
    # new_header = version + patient_id + recording_id + date + fake_time + n_bytes_in_header + reserved +
    # new_header += n_data_records + duration + n_signals
    # new_content = (bytes(new_header, encoding="ascii") + content[256:])

    new_content = header[:168] + bytes(date + fake_time, encoding="ascii") + header[184:] + content[256:]

    # f = open(file_path, 'wb')
    # f.write(new_content)
    # f.close()


# ______________________________________________________________________________________________________________________
def get_patient_info(file_path):
    """ parse sex and age of patient from the patient_id in the header of the edf file
    :param file_path: path of the recording
    :return: sex (0=M, 1=F) and age of patient
    """
    f = open(file_path, 'rb')
    content = f.read()
    f.close()

    header = content[:88]
    patient_id = header[8:88].decode('ascii')
    # the headers "fixed" by tuh nedc data team show a '-' right before the age of the patient. therefore add this to
    # the regex and use the absolute value of the casted age
    [age] = re.findall("Age:(-?\d+)", patient_id)
    [sex] = re.findall("\s\w\s", patient_id)

    sex_id = 0 if sex.strip() == 'M' else 1

    return sex_id, abs(int(age))


# ______________________________________________________________________________________________________________________
def get_recording_length(file_path):
    """ some recordings were that huge that simply opening them with mne caused the program to crash. therefore, open
    the edf as bytes and only read the header. parse the duration from there and check if the file can safely be opened
    :param file_path: path of the directory
    :return: the duration of the recording
    """
    f = open(file_path, 'rb')
    header = f.read(256)
    f.close()
    
    return int(header[236:244].decode('ascii'))


# ______________________________________________________________________________________________________________________
def get_info_with_mne(file_path):
    """ read info from the edf file without loading the data. loading data is done in multiprocessing since it takes
    some time. getting info is done before because some files had corrupted headers or weird sampling frequencies
    that caused the multiprocessing workers to crash. therefore get and check e.g. sampling frequency and duration
    beforehand
    :param file_path: path of the recording file
    :return: file name, sampling frequency, number of samples, number of signals, signal names, duration of the rec
    """
    try:
        edf_file = mne.io.read_raw_edf(file_path, verbose='error')
    except ValueError:
        return None, None, None, None, None, None
        # fix_header(file_path)
        # try:
        #     edf_file = mne.io.read_raw_edf(file_path, verbose='error')
        #     logging.warning("Fixed it!")
        # except ValueError:
        #     return None, None, None, None, None, None

    # some recordings have a very weird sampling frequency. check twice before skipping the file
    sampling_frequency = int(edf_file.info['sfreq'])
    if sampling_frequency < 10:
        sampling_frequency = 1 / (edf_file.times[1] - edf_file.times[0])
        if sampling_frequency < 10:
            return None, sampling_frequency, None, None, None, None

    n_samples = edf_file.n_times
    signal_names = edf_file.ch_names
    n_signals = len(signal_names)
    # some weird sampling frequencies are at 1 hz or below, which results in division by zero
    duration = n_samples / max(sampling_frequency, 1)

    # TODO: return rec object?
    return edf_file, sampling_frequency, n_samples, n_signals, signal_names, duration


# ______________________________________________________________________________________________________________________
def load_data_with_mne(rec):
    """ loads the data using the mne library
    :param rec: recording object holding all necessary data of an eeg recording
    :return: a pandas dataframe holding the data of all electrodes as specified in the rec object
    """
    rec.raw_edf.load_data()
    signals = rec.raw_edf.get_data()

    data = pd.DataFrame(index=range(rec.n_samples), columns=rec.signal_names)
    for electrode_id, electrode in enumerate(rec.signal_names):
        data[electrode] = signals[electrode_id]

    # TODO: return rec object?
    return data


# TODO: try pandas
# ______________________________________________________________________________________________________________________
def write_hdf5(features, in_dir, cmd_args):
    """ writes features to hdf5 file
    :param features: a matrix holding a feature vector for every recordings (maybe someday for every time window of
    every recording)
    :param in_dir: input directory used to extract the name if the class
    :param cmd_args: used to include important information in the file name s.a. window, windowsize etc
    :return: the name of the feature file
    """
    data_set = in_dir.split('/')[-2]
    file_name = os.path.join(cmd_args.output, data_set,
                             '_'.join([data_set, '-'.join([cmd_args.window,
                                                           str(cmd_args.windowsize)+'s',
                                                           str(cmd_args.overlap)+'%']),
                                       cmd_args.bands.replace(',', '-')])) + '.hdf5'

    logging.info("\t\tWriting features to {}.\n".format(file_name))

    hdf5_file = h5py.File(file_name, 'w')
    hdf5_file.create_dataset('data', features.shape, data=features)
    hdf5_file.close()

    return file_name


# ______________________________________________________________________________________________________________________
def write_recording_names(output, inputs, recording_names):
    """ write the names of preprocessed recordings to file.
    :param output: the output directory
    :param inputs: the input directory that is used to extract the name of the class
    :param recording_names: a dictionary holding a list of recording names for every class
    """
    for in_dir in inputs:
        data_set = in_dir.split('/')[-2]
        file_name = os.path.join(output, data_set, "recordings.list")
        with open(file_name, 'w') as f:
            f.writelines('\n'.join(recording_names[in_dir]))


# ______________________________________________________________________________________________________________________
def read_recording_names(feature_files):
    """ reads the names of all recordings of which features were computed from file. this is used to allow backtracking
    of misclassified files after cv
    :param feature_files: list of feature file names. in the same directory exists a list that holds the recording names
    :return: dictionary holding a list of file names for every class
    """
    all_recording_names = {}
    for feature_file in feature_files:
        in_dir = '/'.join(feature_file.split('/')[:-1])
        name = "recordings.list"
        path = os.path.join(in_dir, name)
        if not os.path.exists(path):
            return None
        else:
            with open(path, 'r') as f:
                recording_names = f.readlines()
                recording_names = [recording_name.strip() for recording_name in recording_names]
                # feature file at this position is basically incorrect. it takes the name of the hdf5 file as a key in
                # the dictionary. the directory would be more intuitive. however, cmd_args.input is used later on to
                # access the names which is why this works
                all_recording_names.update({feature_file: recording_names})

    return all_recording_names


# ______________________________________________________________________________________________________________________
def write_feature_labels(output, feature_labels):
    """ writes the feature of the computed features to file
    :param output: output directory
    :param feature_labels: a list of labels that identify the features
    """
    with open(os.path.join(output, 'features.list'), 'w') as out_file:
        out_file.write('\n'.join(feature_labels))


# ______________________________________________________________________________________________________________________
def read_feature_labels(output):
    """ reads the list of features that were computed in the preprocessing step from file. this is used to allow picking
    subsets of these features for training
    :param output: directory where the list was stored to
    :return: ndarray of the feature labels
    """
    path = os.path.join(output, 'features.list')
    if not os.path.exists(path):
        logging.warning("Cannot read feature labels. Path/File does not exist.")
        return None
    else:
        with open(path, 'r') as in_file:
            feature_labels = in_file.readlines()
        feature_labels = [feature_label.strip() for feature_label in feature_labels]

    return np.asarray(feature_labels)


# ______________________________________________________________________________________________________________________
def read_hdf5(file_path):
    """ read the feature file with the specified file path
    :param file_path: path of the file
    :return: ndarray of the data of that file
    """
    if not os.path.exists(file_path):
        logging.fatal("Cannot read feature file {}.".format(file_path))
        exit()
    hdf5_file = h5py.File(file_path, 'r')
    data = np.array(hdf5_file['data'])
    hdf5_file.close()

    return data


# ______________________________________________________________________________________________________________________
def check_out(out_dir, in_dirs=None):
    """ check if the specified output directory exists. if not, create it as well as a subdirectory for all classes
    :param out_dir: output directory
    :param in_dirs: input directory holding the name of the classes
    """
    # create output directory
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

        # for every class create a subdirectory
        for in_dir in in_dirs:
            path = os.path.join(out_dir, in_dir.split('/')[-2])
            if not os.path.exists(path):
                os.makedirs(path)


# ______________________________________________________________________________________________________________________
def write_results(location, classes, results):
    """ writes the results to file. this is mean accuracy, std accuracy, mean precision, std precision, mean recall,
    std recall, mean f1 and std f1 averaged over all folds
    :param location: the output directory
    :param classes: list of feature file paths
    :param results: results as a list of lists. inner lists are accuracies, mean accuracy, std accuracy, ...
    """
    check_out(location)
    # these are tuples consisting of list of values, mean of the values and std the values
    [accuracies, precisions, recalls, f1scores] = results

    file_name = []
    for class_ in classes:
        if class_.endswith('.hdf5'):
            # file_name.append(re.findall('\d/(.*).hdf5$', class_)[0])
            file_name.append(re.findall('([%.\w-]*).hdf5$', class_)[0])

    with open(os.path.join(location, '-'.join(file_name) + '.results'), 'w') as out_file:
        for class_id, class_ in enumerate(classes):
            out_file.write(file_name[class_id].split('_')[0] + '\n')
            # out_file.write(infos)
            # out_file.write(' '.join([str(header_item) for header_item in headers[class_id]]))
            out_file.write('\n\n')

        out_file.write("Mean accuracy is {:.2f} with a std of ({:.2f}).\n".format(accuracies[1], accuracies[2]))
        out_file.write('\n'.join([str(accuracy) for accuracy in accuracies[0]]) + '\n')

        if len(classes) == 2:
            out_file.write("\nMean precision is {:.2f} with a std of ({:.2f}).\n".format(precisions[1], precisions[2]))
            out_file.write('\n'.join([str(precision) for precision in precisions[0]]) + '\n')

            out_file.write("\nMean recall is {:.2f} with a std of ({:.2f}).\n".format(recalls[1], recalls[2]))
            out_file.write('\n'.join([str(recall) for recall in recalls[0]]) + '\n')

            out_file.write("\nMean f1 score is {:.2f} with a std of ({:.2f}).\n".format(f1scores[1], f1scores[2]))
            out_file.write('\n'.join([str(f1score) for f1score in f1scores[0]]) + '\n')


# ______________________________________________________________________________________________________________________
def write_classifier_output(location, folds, labels, predictions, class_probs, names=None):
    """ writes the classifier cv predictions to file. that is, for every fold store the recordings that were predicted,
    their label and prediction and the probabilities to one row.
    :param location: directory where to save the output
    :param folds: number of cv folds
    :param labels: the labels of the predicted recordings
    :param predictions: the predictions of the recordings that should equal the labels
    :param class_probs: the probabilities of the different classes
    :param names: the names of the predicted recordings
    """
    with open(os.path.join(location, '-'.join(["classifier", "fold", "predictions"]) + '.txt'), 'w') as out_file:
        for fold in range(folds):
            out_file.write("fold " + str(fold+1) + ':\n')
            out_file.write("{:50} {:<12} {:<12} {:<9} {:<9}\n".format("recording", "prediction", "label", "class 0",
                                                                      "class 1"))
            fold_labels, fold_predictions, fold_class_probs = labels[fold], predictions[fold], class_probs[fold]

            if names is not None and len(names) != 0:
                fold_names = np.hstack(names[fold])
            else:
                fold_names = len(fold_predictions) * ['']

            for pred_lab_tuple in zip(fold_names, fold_predictions, fold_labels, fold_class_probs[:, 0],
                                      fold_class_probs[:, 1]):
                (name, pred, label, prob1, prob2) = pred_lab_tuple
                out_file.write("{:50} {:<12} {:<12} {:<9.2f} {:<9.2f}\n".format(name, pred, label, prob1, prob2))
            out_file.write('\n')


# ______________________________________________________________________________________________________________________
def read_cv_predictions(output):
    """ this is used to read the cv predictions created in 'write_classifier_output'. they can then be used to identify
    misclassified recordings and be a starting point for further analysis
    :param output: the directory where the fold predictions are stored
    :return: ndarrays that hold the recording names, predictions, labels and probabilities of both classes
    """
    with open(os.path.join(output, "classifier-fold-predictions.txt"), 'r') as in_file:
        lines = in_file.readlines()

    lines = ''.join(lines)
    # there is always a new line at the end of the file. strip it
    fold_predictions = lines.split('\n\n')[:-1]

    # parse the lines of the file
    names, preds, all_labels, all_prob0s, all_prob1s = [], [], [], [], []
    for fold_prediction in fold_predictions:
        fold_prediction = fold_prediction.split('\n')

        rec_names, predictions, labels, prob0s, prob1s = [], [], [], [], []
        for line in fold_prediction[2:]:
            [name, prediction, label, prob0, prob1] = re.findall('[\w_./]+', line)

            rec_names.append(name)
            predictions.append(int(prediction))
            labels.append(int(label))
            prob0s.append(float(prob0))
            prob1s.append(float(prob1))

        names.append(rec_names)
        preds.append(predictions)
        all_labels.append(labels)
        all_prob0s.append(prob0s)
        all_prob1s.append(prob1s)

    return np.asarray(names), np.asarray(preds), np.asarray(all_labels), np.asarray(all_prob0s), np.asarray(all_prob1s)


# ______________________________________________________________________________________________________________________
def write_to_google_result_tracking(classes, n_recs, window, w_size, t_overlap, clf, n_est, feats, elecs, N,
                                    bands, f_overlap, scale, folds, autoskl, results, notes):
    """ automatically update the goolge spreadsheet that tracks the results """
    outh_file = "client_secret_774608351368-1rhj46mkk760tjrd5k40u82d89uqd9vc.apps.googleusercontent.com.json"
    if not os.path.exists(outh_file):
        logging.warning("Cannot upload the data to google result_tracking spreadsheet. Maybe you are not authorized?")
        return

    [accs, precs, recs, f1s] = results
    gc = pygsheets.authorize(outh_file=outh_file)
    sh = gc.open("result_tracking")

    # a row looks like this:
    # classes: class_0, class_1
    # recs: n_recs_class_0, n_recs_class_1
    # err: n_err_class_0, n_err_class_1
    # -
    # window: the window function being used
    # size: the size of the time window
    # overlap: overlap of time windows
    # -
    # clf: the classifier being used
    # est: number of estimators
    # -
    # features: a list of features being used
    # elecs: a list of electrodes being used
    # N: the number of features being used
    # bands: the frequency bands being used
    # overlap: overlap of frequency bands
    # scale/norm: whether and how features are scaled / normalized
    # -
    # folds: the number of cv folds
    # mean acc:
    # std acc:
    # mean prec:
    # std prec:
    # mean rec:
    # std rec:
    # mean f1:
    # std f1:
    classes = [class_.split('/')[-1].split('_')[0].split('v')[0] for class_ in classes]
    classes = ', '.join(classes)
    n_recs = ', '.join([str(n_rec) for n_rec in n_recs])
    w_size = str(w_size)
    t_overlap = str(t_overlap)
    clf = str(clf)
    n_est = str(n_est)
    feats = ', '.join(feats)
    elecs = ', '.join(elecs)
    N = str(N)
    bands = ', '.join(bands.split(','))
    f_overlap = str(f_overlap)
    autoskl_str = [str(auto) for auto in autoskl]
    autoskl_str = ', '.join(autoskl_str)
    mean_accs = "{:.2f}".format(accs[1])
    std_accs = "{:.2f}".format(accs[2])
    mean_precs = "{:.2f}".format(precs[1])
    std_precs = "{:.2f}".format(precs[2])
    mean_recs = "{:.2f}".format(recs[1])
    std_recs = "{:.2f}".format(recs[2])
    mean_f1s = "{:.2f}".format(f1s[1])
    std_f1s = "{:.2f}".format(f1s[2])

    if autoskl[0] == 0 and autoskl[1] == 0:
        wks = sh.worksheet_by_title("auto_normal_abnormalv1.1.0")
        wks.append_table(values=[classes, n_recs, window, w_size, t_overlap, clf, n_est, feats, elecs, N, bands,
                                 f_overlap, scale, folds, mean_accs, std_accs, mean_precs, std_precs,
                                 mean_recs, std_recs, mean_f1s, std_f1s, notes])
    else:
        wks = sh.worksheet_by_title("auto-skl_normal_abnormalv1.1.0")
        wks.append_table(values=[classes, n_recs, window, w_size, t_overlap,  clf, n_est,  feats, elecs, N,
                                 bands, f_overlap, scale, folds, autoskl_str, mean_accs, std_accs, mean_precs,
                                 std_precs, mean_recs, std_recs, mean_f1s, std_f1s, notes])


# ______________________________________________________________________________________________________________________
def update_cmd_args(cmd_args, feature_files):
    """ updates the cmd arguments s.t. the automatic entries of google spreadsheet are correct. i.e. when running the
    pipeline with 'no-pre' option, it would use the default values for window, window_size etc. this function overrides
    these values with the actual values as given by the feature file name
    :param cmd_args: dictionary of cmd arguments
    :param feature_files: list of feature file paths
    :return: updated cmd arguemnts including the information stored in feature file name
    """
    cmd_args.input = feature_files

    windows, window_sizes, overlaps, all_bands = [], [], [], []
    for feature_in_file in cmd_args.input:
        file_name = feature_in_file.split('/')[-1]
        [dummy, window_info, band_info] = file_name.split('_')
        band_info = band_info.split('.')[0]
        window, window_size, overlap = window_info.split('-')
        window_size = window_size[:-1]
        overlap = overlap[:-1]
        bands = band_info.replace('-', ',')

        windows.append(window)
        window_sizes.append(window_size)
        overlaps.append(overlap)
        all_bands.append(bands)

    if not (len(np.unique(windows)) == 1 and
            len(np.unique(window_sizes)) == 1 and
            len(np.unique(overlaps)) == 1 and
            len(np.unique(all_bands)) == 1):
        logging.error("You are trying to train on features that were computed with unequal window function, window"
                      "size, overlap or frequency bands.")
        exit()

    cmd_args.window = windows[0]
    cmd_args.windowsize = window_sizes[0]
    cmd_args.overlap = overlaps[0]
    cmd_args.bands = all_bands[0]

    return cmd_args
