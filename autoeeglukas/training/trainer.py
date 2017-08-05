from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.preprocessing import Normalizer, StandardScaler, MaxAbsScaler
from sklearn.model_selection import KFold, train_test_split
from sklearn.decomposition import PCA
import numpy as np
import logging
import re

from visualization import plotting
from utils import my_io

SEED = 4316932


# TODO: if features were computed per window, adapt cross validation and evaluation
class Trainer(object):
    """
    """

# ______________________________________________________________________________________________________________________
    def introduce(self):
        # TODO: log the electrodes and features chosen by cmd argument
        logging.info("\t\tHi, I am the trainer! I am part of the pipeline and I try to find a classifier that best "
                     "fits your task. I use a cross-validation approach with {} folds and a time-based data split. "
                     "I utilize {} features that were computed in the preprocessing step."
                     .format(self.folds, ', '.join(['patient', 'fft', 'time', 'pyeeg', 'dwt'])))

# ______________________________________________________________________________________________________________________
    def build_feature_vector(self, out_dir, feature_files):
        """ Reads the features from files and arranges them.
        :param out_dir: the directory where the hdf5 files were originally written to
        :param feature_files: a list of hdf5 files containing the features of the different classes
        :param feats: a list of subsets of features that should be used for classification
        :return: an array with the shape n_classes x n_recordings x n_features
        """
        if self.feature_labels is None:
            self.feature_labels = my_io.read_feature_labels(out_dir)

        features = []
        for feature_file in feature_files:
            logging.info("\t\t\tReading feature file: {}".format(feature_file))
            data = my_io.read_hdf5(feature_file)
            features.append(data)

        return np.asarray(features)

# ______________________________________________________________________________________________________________________
    def take_subset_of_feats(self, features_list, subsets, split_id):
        # the labels are arranged as follows: <domain>_<feature>_<freq band>_<electrode>
        if 'all' in subsets:
            return features_list
        else:
            # if doing electrodes make sure the given query matches the internal name convention (upper case)
            if split_id == -1:
                subsets = [subset.upper() for subset in subsets]
            new_feature_label_ids = []
            for subset in subsets:
                for feature_label_id, feature_label in enumerate(self.feature_labels):
                    if split_id is not None:
                        if subset in feature_label.split('_')[split_id]:
                            new_feature_label_ids.append(feature_label_id)
                    else:
                        if subset in feature_label:
                            new_feature_label_ids.append(feature_label_id)

            # only pick those columns from the features that correspond to the subset of features / electrodes
            if not new_feature_label_ids:
                return features_list

            new_feature_label_ids = np.asarray(sorted(new_feature_label_ids))
            for class_ in range(len(features_list)):
                features_list[class_] = features_list[class_][:, new_feature_label_ids]

            # also adapt the features_labels
            self.feature_labels = self.feature_labels[new_feature_label_ids]
            return features_list

# ______________________________________________________________________________________________________________________
    def normalize_features(self, train, test):
        """ Normalize the feature values to avoid a bias. Therefore, first fit the normalizer by feature and afterwards
        fit a second normalizer by sample.
        :param train:
        :param test:
        :return:
        """
        normalizer1 = Normalizer()
        normalizer2 = Normalizer()
        train = normalizer1.fit_transform(train.T).T
        test = normalizer1.transform(test.T).T
        # train = normalizer2.fit_transform(train)
        # test = normalizer2.transform(test)
        # print(train[:, 0])

        return train, test

# ______________________________________________________________________________________________________________________
    def reduce_dimensions(self, train, test):
        """
        :param train:
        :param test:
        :return:
        """
        # print(train.shape, test.shape)
        pca = PCA(n_components=8)
        pca.fit(train)

        train = pca.transform(train)
        test = pca.transform(test)
        # print(train.shape, test.shape)

        # print(pca.explained_variance_)
        # print(pca.explained_variance_ratio_)
        return train, test

# ______________________________________________________________________________________________________________________
    def scale_features(self, train, test):
        """ scales the features by their absolute maximum value. scaling is fit on train fold and afterwards also
        applied to test fold
        :param train: train fold
        :param test: test fold
        :return: scaled train and test fold
        """
        scaler = MaxAbsScaler(copy=False)
        train = scaler.fit_transform(train)
        test = scaler.transform(test)

        return train, test, str(scaler)

# ______________________________________________________________________________________________________________________
    def get_train_test(self, fold, features, train_folds, test_folds):
        """ for every fold take the corresponding per class train subsets and concat them. also create the labels. do
        the same for test subsets
        :param fold: the fold_id
        :param features: shape gives number of classes
        :param train_folds: the list of train folds of the individual classes
        :param test_folds: the list of test folds of the individual classes
        :return: train folds of all classes of the specified fold, labels for training, test folds of all classes of the
        specified fold, labels for testing
        """
        train, labels_train, test, labels_test = [], [], [], []
        for class_id, class_ in enumerate(features):
            curr_train = train_folds[class_id][fold]
            train.append(curr_train)
            labels_train += len(curr_train) * [class_id]

            curr_test = test_folds[class_id][fold]
            test.append(curr_test)
            labels_test += len(curr_test) * [class_id]

        if len(train[0].shape) == 1:
            for train_fold in range(len(train)):
                train[train_fold] = train[train_fold].reshape(-1, 1)
            for test_fold in range(len(test)):
                test[test_fold] = test[test_fold].reshape(-1, 1)

        # if getting folds of recording names the shape is incorrect, e.g. (21, ).. reshape to (21, 1)
        return np.vstack(train), labels_train, np.vstack(test), labels_test

# ______________________________________________________________________________________________________________________
    def evaluate_predictions(self, labels_test, predictions, n_classes):
        """ use sklearn metrics to evaluate the CV
        :param labels_test:
        :param predictions:
        :param n_classes:
        """
        self.accs.append(100 * accuracy_score(labels_test, predictions))

        if n_classes == 2:
            self.precs.append(100 * precision_score(labels_test, predictions))
            self.recs.append(100 * recall_score(labels_test, predictions))
            self.f1s.append(100 * f1_score(labels_test, predictions))

# ______________________________________________________________________________________________________________________
    def majority_vote(self, predictions, window_counts):
        # TODO: fold the window counts according to the test folds, s.t. majority vote is possible
        print(predictions)
        print(window_counts)
        return predictions

# ______________________________________________________________________________________________________________________
    def auto_sklearn(self, output, in_files, features, rec_names):
        """ uses the autosklearn module to optimize the score
        :param output: the output directory
        :param in_files: a list of feature matrices of the different classes
        :param features: the features that were computed in preprocessing. shape n_classes x n_recordings x n_features
        :param rec_names: a list of all recordings that were used to compute features
        """
        import autosklearn.classification
        from autosklearn import metrics
        y = []
        for class_id, class_ in enumerate(features):
            y.extend(len(class_) * [class_id])

        x = np.vstack((features[0], features[1]))
        x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=SEED)

        # make sure these are ndarrays
        x_train, x_test = np.asarray(x_train), np.asarray(x_test)
        y_train, y_test = np.asarray(y_train), np.asarray(y_test)

        rec_names = rec_names[in_files[0]] + rec_names[in_files[1]]

        # set up auto-sklearn
        clf = autosklearn.classification.AutoSklearnClassifier(
            time_left_for_this_task=self.total_budget,
            per_run_time_limit=self.run_budget,

            # include_estimators=["random_forest", ],
            # include_preprocessors=["no_preprocessing", ],
            # ensemble_size=5,
            # initial_configurations_via_metalearning=0,

            tmp_folder='/'.join([output, 'autoskl', 'tmp']),
            delete_tmp_folder_after_terminate=False,
            output_folder='/'.join([output, 'autoskl', 'out'], ),
            delete_output_folder_after_terminate=False,

            resampling_strategy='cv',
            resampling_strategy_arguments={'folds': self.folds},

            seed=SEED,
        )

        # acc_scorer = autosklearn.metrics.make_scorer(name="acc",
        #                                              score_func="accuracy",
        #                                              greater_is__better=True,
        #                                              needs_proba=True,
        #                                              needs_threshold=False)
        # clf.fit(x_train.copy(), y_train.copy(), metric=acc_scorer)
        clf.fit(x_train.copy(), y_train.copy())
        print("cv results", clf.cv_results_)

        # prec_scorer = autosklearn.metrics.make_scorer(name="prec",
        #                                              score_func="precision",
        #                                              greater_is__better=True,
        #                                              needs_proba=True,
        #                                              needs_threshold=False)
        # clf.fit(x_train.copy(), y_train.copy(), metric=prec_scorer)
        # print(clf.cv_results_["mean_test_score"])

        clf.refit(x_train.copy(), y_train.copy())
        predictions = clf.predict(x_test)
        print("Accuracy score", accuracy_score(y_test, predictions))
        print("Accuracy score {:g} using {:s}".
              format(accuracy_score(y_test, predictions), clf._automl._automl._metric.name))

        print("models", clf.show_models())

        prediction_probs = clf.predict_proba(x_test)
        self.evaluate_predictions(y_test, predictions, len(features))
        my_io.write_classifier_output(output, 1, [y_test], [predictions], [prediction_probs], [rec_names])

        # TODO: return a lot of stuff. clf.show_models(), clf.cv_results_["mean_test_score"]
        return clf

# ______________________________________________________________________________________________________________________
    def cross_validate(self, output, features, window_counts, train_folds, test_folds,
                       rec_names_train_folds=None, rec_name_test_folds=None):
        """ Folds are available per class. Concats them to train and test sets. A standard scaler is fit to the train
        data and afterwards applied to the according test data. The classifier is then fit to the training data and
        evaluated on the test data. accuracies, precisions and recalls of the folds are stored and returned.
        :param output: output directory
        :param features: all features that will be split into individual folds
        :param window_counts: window counts. can be used to combine window predictions
        :param train_folds: a list of train folds
        :param test_folds: a list of test folds
        """
        clf = RandomForestClassifier(
            n_jobs=self.jobs,
            n_estimators=self.estimators,
            random_state=SEED,
        )

        all_labels_test, all_predictions, all_prediction_probs, all_rec_names_test, all_importances = [], [], [], [], []
        for fold in range(self.folds):
            train, labels_train, test, labels_test = self.get_train_test(fold, features, train_folds, test_folds)

            # assemble the file names to go with the predictions
            if rec_names_train_folds is not None and rec_name_test_folds is not None:
                rec_names_train, dummy, rec_names_test, dummy = self.get_train_test(fold, features,
                                                                                    rec_names_train_folds,
                                                                                    rec_name_test_folds)
                all_rec_names_test.append(rec_names_test)

            # normalize/standardize/scale features?
            train, test, self.scaler = self.scale_features(train, test)
            # train, test = self.normalize_features(train, test)

            # reduce dimensions with pca?
            # train, test = self.reduce_dimensions(train, test)

            # train the classifier on the train subset and evaluate it on the test subset
            clf.fit(train, labels_train)
            predictions = clf.predict(test)
            prediction_probs = clf.predict_proba(test)
            importances = clf.feature_importances_ * 100

            # TODO: calculate the out of bag error and use it for visualization?
            # oob_error = 1-rf.oob_score_

            # if features were generated per window, the window predictions should be combined with e.g. a majority vote
            if not self.perrec:
                predictions = self.majority_vote(predictions, window_counts)

            self.evaluate_predictions(labels_test, predictions, len(features))

            all_labels_test.append(labels_test)
            all_predictions.append(predictions)
            all_prediction_probs.append(prediction_probs)
            all_importances.append(importances)

        self.importances = np.asarray(all_importances)
        my_io.write_classifier_output(output, self.folds, all_labels_test, all_predictions, all_prediction_probs,
                                      all_rec_names_test)

        return clf

    # TODO: when there is one sample per time window, see the time windows as a unit. don't split them?
# ______________________________________________________________________________________________________________________
    def create_folds(self, features):
        """ Features are available per class. Splits the data into folds subsets for cross validation.
        :param features:
        :return: list of train folds and list of test folds, that each hold a list for the different classes

        """
        # split the features of all classes in self.folds subsets for training and validation
        kf = KFold(n_splits=self.folds)
        train_folds, test_folds = [], []

        # features is in shape n_classes x n_recordings x n_features
        for class_ in features:
            class_train_folds, class_test_folds = [], []
            for train, test in kf.split(class_):
                class_train_folds.append(class_[train])
                class_test_folds.append(class_[test])

            train_folds.append(class_train_folds)
            test_folds.append(class_test_folds)

        return train_folds, test_folds

# TODO: remove feature files. it is not needed. replace by cmd_args.input
# ______________________________________________________________________________________________________________________
    def train(self, cmd_args, window_counts):
        """ take the features computed in the preprocessing step. split them into train and test folds to perform CV.
        :param cmd_args: output directory and visualization is read from cmd_args
        :param feature_files: the list of feature files of the individual classes
        :param window_counts: if features were generated per window instead of per channel, the window counts are needed
        to combine the window predictions to channel and to recording predictions
        """
        # shape of feature_list: n_classes X n_recordings x n_features
        features_list = self.build_feature_vector(cmd_args.output, cmd_args.input)
        # TODO: check what happens if feature labels cannot be read
        # picks the subsets of the feature matrices as specified by the domain cmd argument
        features_list = self.take_subset_of_feats(features_list, cmd_args.domain, 0)
        # picks the subsets of the feature matrices as specified by the feats cmd argument
        features_list = self.take_subset_of_feats(features_list, cmd_args.feats, None)
        # picks the subsets of the feature matrices as specified by the elecs cmd argument
        features_list = self.take_subset_of_feats(features_list, cmd_args.elecs, -1)

        # these are the names of the recordings. they are added to the fold predictions, s.t. it allows for a
        # traceback of misclassified recordings
        rec_names = my_io.read_recording_names(cmd_args.input)

        # check if autosklearn should be done or traditional cv?
        if cmd_args.auto_skl[0] == 0 and cmd_args.auto_skl[1] == 0:
            # TODO: this can already be done at the very beginning when reading files from directory and checking
            # whether they are fine-ish
            # check if there is enough data to do cv
            for class_id, class_ in enumerate(features_list):
                if len(class_) < self.folds:
                    logging.fatal("Number of samples in {} is too small.".format(cmd_args.input[class_id]))
                    exit()

            train_folds, test_folds = self.create_folds(features_list)
            if not self.perrec:
                trash, window_counts = self.create_folds(window_counts)

            rec_names_train_folds, rec_names_test_folds = None, None
            if rec_names is not None:
                # print(len(rec_names[cmd_args.input[0]]), type(rec_names[cmd_args.input[0]]))
                # print(len(rec_names[cmd_args.input[1]]), type(rec_names[cmd_args.input[1]]))
                # print(rec_names[cmd_args.input[0]], rec_names[cmd_args.input[1]])
                l1 = np.asarray(rec_names[cmd_args.input[0]])
                l2 = np.asarray(rec_names[cmd_args.input[1]])
                rec_names_new = np.array([l1, l2])

                rec_names_train_folds, rec_names_test_folds = self.create_folds(rec_names_new)

            clf = self.cross_validate(cmd_args.output, features_list, window_counts, train_folds, test_folds,
                                      rec_names_train_folds, rec_names_test_folds)

            if 'all' in cmd_args.visualize or 'train' in cmd_args.visualize:
                plotting.plot_cv_results(cmd_args.output, self.folds, self.get_results())
                plotting.plot_feature_importances(self.importances, cmd_args.output, self.feature_labels)
                plotting.plot_feature_importances_spatial(self.importances, cmd_args.output, self.feature_labels)

            # TODO: add freq band overlap?
            if not cmd_args.no_up:
                my_io.write_to_google_result_tracking(cmd_args.input, [len(features_list[0]), len(features_list[1])],
                                                      cmd_args.window, cmd_args.windowsize, cmd_args.overlap,
                                                      ''.join([ch for ch in str(clf).split('(')[0] if ch.isupper()]),
                                                      self.estimators, cmd_args.feats, cmd_args.elecs,
                                                      len(self.feature_labels), cmd_args.bands, 50,
                                                      self.scaler.split('(')[0], self.folds,
                                                      [self.total_budget, self.run_budget], self.get_results(),
                                                      ' '.join(re.split('\s{2,}', str(clf))))

        else:
            clf = self.auto_sklearn(cmd_args.output, cmd_args.input, features_list, rec_names)
            if 'all' in cmd_args.visualize or 'train' in cmd_args.visualize:
                # the 1 should be folds. need to adapt autosklearn to do self.folds cv
                plotting.plot_cv_results(cmd_args.output, 1, self.get_results())

            # TODO: add freq band overlap?, replace '?'
            if not cmd_args.no_up:
                my_io.write_to_google_result_tracking(cmd_args.input, [len(features_list[0]), len(features_list[1])],
                                                      cmd_args.window, cmd_args.windowsize, cmd_args.overlap,
                                                      '?', '?', '?', '?', '?', cmd_args.bands, 50,
                                                      '?', '?', [self.total_budget, self.run_budget],
                                                      self.get_results(), ' '.join(re.split('\s{2,}', str(clf))))

        # write and visualize the results of cv
        my_io.write_results(cmd_args.output, cmd_args.input, self.get_results())


# ______________________________________________________________________________________________________________________
    def get_results(self):
        return (self.accs, np.mean(self.accs), np.std(self.accs)), \
               (self.precs, np.mean(self.precs), np.std(self.precs)), \
               (self.recs, np.mean(self.recs), np.std(self.recs)), \
               (self.f1s, np.mean(self.f1s), np.std(self.f1s))

# ______________________________________________________________________________________________________________________
    def __init__(self, folds=10, jobs=1, estimators=64, perrec=True, auto_skl=[0, 0]):
        self.perrec = perrec
        self.folds = folds
        self.jobs = jobs
        self.estimators = estimators
        self.scaler = None

        self.accs = []
        self.precs = []
        self.recs = []
        self.f1s = []

        self.feature_labels = None
        self.importances = []

        self.total_budget = auto_skl[0]
        self.run_budget = auto_skl[1]

        self.introduce()
