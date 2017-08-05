#!/usr/bin/env python3.5
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection
import numpy as np
import argparse

from visualization import plotting
from utils import my_io


RANDOM_SEED = 345738382
np.random.seed(RANDOM_SEED)


def equal_class_splits(features_a, features_b, labels_a, labels_b):
    random_state = np.random.randint(RANDOM_SEED)

    # TODO: this can be combined in a loop
    x_train1, x_test1, y_train1, y_test1 = model_selection.train_test_split(features_a, labels_a,
                                                                            random_state=random_state, test_size=0.33)
    x_train2, x_test2, y_train2, y_test2 = model_selection.train_test_split(features_b, labels_b,
                                                                            random_state=random_state, test_size=0.33)

    if len(x_train1.shape) == 1:
        x_train1 = x_train1.reshape(-1, 1)
        x_train2 = x_train2.reshape(-1, 1)
        x_test1 = x_test1.reshape(-1, 1)
        x_test2 = x_test2.reshape(-1, 1)

    x_train = np.vstack((x_train1, x_train2))
    x_test = np.vstack((x_test1, x_test2))
    y_train = y_train1 + y_train2
    y_test = y_test1 + y_test2

    return x_train, x_test, y_train, y_test


def time_based_splits(features_a, features_b, labels_a, labels_b,):
    # since data was read with myio and the "time key", the last third of samples is used for evaluation
    splitpoint_a = int(len(features_a) * 2 / 3)
    splitpoint_b = int(len(features_b) * 2 / 3)

    x_train = np.vstack((features_a[:splitpoint_a], features_b[:splitpoint_b]))
    x_test = np.vstack((features_a[splitpoint_a:], features_b[splitpoint_b:]))

    # print(x_train.shape)
    # if len(x_train.shape) == 1:
    #     x_train = x_train.reshape(-1, 1)
    #     x_test = x_test.reshape(-1, 1)
    # print(x_train.shape)

    y_train = labels_a[:splitpoint_a] + labels_b[:splitpoint_b]
    y_test = labels_a[splitpoint_a:] + labels_b[splitpoint_b:]

    return x_train, x_test, y_train, y_test


def train(x, y):
    accuracy = 0
    best_thresh = 0
    label_order = []
    for thresh_id, thresh in enumerate(x):
        y_pred = thresh_id * [True] + (len(y) - thresh_id) * [False]

        curr_accuracy = accuracy_score(y, y_pred)
        if curr_accuracy > accuracy:
            accuracy = curr_accuracy
            best_thresh = thresh
            label_order = y_pred

    for thresh_id, thresh in enumerate(x):
        y_pred = thresh_id * [False] + (len(y) - thresh_id) * [True]

        curr_accuracy = accuracy_score(y, y_pred)
        if curr_accuracy > accuracy:
            accuracy = curr_accuracy
            best_thresh = thresh
            label_order = y_pred

    return best_thresh, [label_order[0], label_order[-1]]


def validate(x, y, label_order, thresh):
    # find value closest to thresh and use it to split into pos and neg samples
    split_point = x.index(min(x, key=lambda e: abs(e - thresh)))
    labels = split_point * [label_order[0]] + (len(y) - split_point) * [label_order[1]]

    accuracy = accuracy_score(labels, y)
    precision = precision_score(labels, y)
    recall = recall_score(labels, y)

    return accuracy, precision, recall


def cross_validate(features_a, features_b, labels_a, labels_b, folds):
    thresholds, accuracies, precisions, recalls = [], [], [], []
    for fold in range(folds):
        # x_train, x_test, y_train, y_test = equal_class_splits(features_a, features_b, labels_a, labels_b)
        x_train, x_test, y_train, y_test = time_based_splits(features_a, features_b, labels_a, labels_b)

        x_train, y_train = zip(*sorted(zip(x_train, y_train)))
        x_test, y_test = zip(*sorted(zip(x_test, y_test)))

        threshold, label_order = train(x_train, y_train)
        accuracy, precision, recall = validate(x_test, y_test, label_order, threshold)

        thresholds.append(threshold)
        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)

    return accuracies, thresholds, precisions, recalls


def main(cmd_args):
    folds = 10

    features_a = my_io.read_hdf5(cmd_args.input1)
    features_b = my_io.read_hdf5(cmd_args.input2)

    # get the header and the data from pandas data frame
    header_a = features_a.columns
    header_b = features_b.columns
    features_a = features_a.as_matrix()
    features_b = features_b.as_matrix()

    labels_a = len(features_a) * [True]
    labels_b = len(features_b) * [False]

    # if only one band should be used for best threshold classification
    if cmd_args.band is not None:
        band = cmd_args.band
        features_a = features_a[:, band]
        features_b = features_b[:, band]

        accuracies, thresholds, precisions, recalls = cross_validate(features_a, features_b, labels_a, labels_b, folds)
        plotting.scatter_features([features_a, features_b], np.mean(thresholds), cmd_args.output)

        print('thresholds:', np.mean(thresholds), np.std(thresholds))

    # use all frequency bands and a random forest to classify
    else:
        rf = RandomForestClassifier()
        accuracies, precisions, recalls = [], [], []
        for folds in range(folds):
            # x_train, x_test, y_train, y_test = equal_class_splits(features_a, features_b, labels_a, labels_b)
            x_train, x_test, y_train, y_test = time_based_splits(features_a, features_b, labels_a, labels_b)

            rf = rf.fit(x_train, y_train)
            predictions = rf.predict(x_test)

            accuracies.append(accuracy_score(y_test, predictions))
            precisions.append(precision_score(y_test, predictions))
            recalls.append(recall_score(y_test, predictions))
            # print(classification_report(y_test, predictions))

    print('mean accuracy:', 100 * np.mean(accuracies), 100 * np.std(accuracies))
    print('mean precision:', 100 * np.mean(precisions), 100 * np.std(precisions))
    print('mean recall:', 100 * np.mean(recalls), 100 * np.std(recalls))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('input1', type=str, help='hdf5 file of first class')
    parser.add_argument('input2', type=str, help='hdf5 file of second class')
    parser.add_argument('-o', '--output', type=str, help='output directory', default=None, metavar='')
    parser.add_argument('-b', '--band', type=int, help='frequency band used for classification', metavar='',
                        default=None)

    cmd_arguments, unknown = parser.parse_known_args()
    main(cmd_arguments)
