import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt
from numpy.random import RandomState
import numpy as np
import seaborn
import scipy
import pylab
import logging
import os

from utils import my_io
# TODO: make this a class and give it output location as a member?


# ______________________________________________________________________________________________________________________
def plot_rec_analysis(signal_processing_steps, sampling_freq, cmd_args=None):
    """ Plot raw signal, decomposed by freq band, cleaned signal, decomposed by band
    :param rec:
    :param cmd_args:
    :return:
    """
    # TODO: plot series of time windows to see how the different window functions change the overall signal
    # TODO: also try overlap
    # scale to microvolts
    ch = signal_processing_steps[0] * 1000000
    # take 2 seconds of the signal
    secs = 2

    sart_shift = 20
    ch_max = max(ch[sart_shift*sampling_freq:(sart_shift+secs)*sampling_freq]) * 1.1
    ch_min = min(ch[sart_shift*sampling_freq:(sart_shift+secs)*sampling_freq]) * 1.1

    n_figures = 7
    # this is the cleaned and cut signal
    ch_cleaned = signal_processing_steps[1]
    ch_ft = signal_processing_steps[2]
    spectrum_median = signal_processing_steps[3]
    # print(ch_ft.shape)

    # be aware of the start time shift
    # this is the original signal
    plt.subplot(n_figures, 1, 1)
    plt.plot(np.arange(sampling_freq*secs), ch[60*sampling_freq:(60+secs)*sampling_freq])
    plt.xticks(np.linspace(0, sampling_freq*secs, secs+1), np.arange(secs+1))
    plt.ylim([ch_min, ch_max])
    plt.xlabel('time [s]')
    plt.ylabel('amplitude')
    # plt.title("")
    plt.grid(True)

    # this is the spectrum of the signal
    plt.subplot(n_figures, 1, 2)
    plt.plot(np.arange(len(ch_ft)), np.abs(ch_ft))
    # plt.title("spectrum")
    plt.ylabel('amplitude')
    plt.xlabel("frenquency")
    x_tick_locations = [1, 4, 8, 12, 30, 60, 120]
    x_tick_locations2 = [int(tick/(1/secs)) for tick in x_tick_locations]
    x_tick_labels = [str(tick) for tick in x_tick_locations]
    plt.xticks(x_tick_locations2, x_tick_labels)
    plt.grid(True)

    # this is the cleaned signal
    plt.subplot(n_figures, 1, 3)
    plt.plot(np.arange(sampling_freq*secs), ch_cleaned[:sampling_freq*secs])
    plt.xticks(np.linspace(0, sampling_freq*secs, secs+1), np.arange(secs+1))
    plt.xlabel('time [s]')
    plt.ylabel('amplitude')
    plt.ylim([ch_min, ch_max])
    # plt.title("")
    plt.grid(True)

    # this is the spectrum of the signal
    plt.subplot(n_figures, 1, 4)
    plt.plot(np.arange(len(ch_ft)), np.abs(ch_ft))
    # plt.title("spectrum")
    plt.ylabel('amplitude')
    plt.xlabel("frenquency")
    x_tick_locations = [1, 4, 8, 12, 30, 60, 120]
    x_tick_locations2 = [int(tick/(1/secs)) for tick in x_tick_locations]
    x_tick_labels = [str(tick) for tick in x_tick_locations]
    plt.xticks(x_tick_locations2, x_tick_labels)
    plt.grid(True)

    # this is the zoomed-in spectrum of the signal
    plt.subplot(n_figures, 1, 5)
    zoomed_in = np.abs(ch_ft)[:int(30/(1/secs))]
    plt.plot(np.arange(len(zoomed_in)), zoomed_in)
    # plt.title("spectrum")
    plt.ylabel('amplitude')
    plt.xlabel("frenquency")
    plt.xticks(x_tick_locations2[:5], x_tick_labels[:5])
    plt.grid(True)

    # this is the reconstructed signal
    reconstructed_signal = np.fft.irfft(ch_ft)
    plt.subplot(n_figures, 1, 6)
    plt.plot(np.arange(len(reconstructed_signal)), reconstructed_signal)
    plt.xticks(np.linspace(0, sampling_freq*secs, secs+1), np.arange(secs+1))
    plt.ylabel('amplitude')
    plt.xlabel("frenquency")
    plt.grid(True)
    plt.ylim([ch_min, ch_max])

    # this is the reconstructed signal
    reconstructed_signal = np.fft.irfft(ch_ft)
    plt.subplot(n_figures, 1, 7)
    plt.plot(np.arange(2*len(reconstructed_signal)), np.hstack((reconstructed_signal, reconstructed_signal)))
    plt.xticks(np.linspace(0, sampling_freq*secs*2, 2*secs+1), np.arange(2*secs+1))
    plt.ylabel('amplitude')
    plt.xlabel("frenquency")
    plt.grid(True)
    plt.ylim([ch_min, ch_max])

    # edf_file_name = '/'.join(rec.name.split('/')[-4:])
    # path = os.path.join(cmd_args.output, edf_file_name.replace('edf', 'pdf').replace('/', '_'))
    # plt.savefig(path, bbox_inches="tight")

    plt.show()


# ______________________________________________________________________________________________________________________
def plot_cv_results(output, folds, results):
    # import seaborn
    [accs, precs, recs, f1s] = results

    ind = np.arange(folds)  # the x locations for the groups
    width = 0.2  # the width of the bars

    # colors for plotting
    a = .7
    c1 = "hotpink"
    c2 = "deepskyblue"
    c3 = "chartreuse"
    c4 = "darkorange"

    # cv fold results as bars
    bar1 = plt.bar(ind, accs[0], width, color=c1, alpha=a, yerr=accs[2], error_kw=dict(ecolor='k', alpha=a))
    bar2 = plt.bar(ind + width, precs[0], width, color=c2, alpha=a, yerr=precs[2], error_kw=dict(ecolor='k', alpha=a))
    bar3 = plt.bar(ind + 2*width, recs[0], width, color=c3, alpha=a, yerr=recs[2], error_kw=dict(ecolor='k', alpha=a))
    bar4 = plt.bar(ind + 3*width, f1s[0], width, color=c4, alpha=a, yerr=f1s[2], error_kw=dict(ecolor='k', alpha=a))

    # mean cv results as hlines
    hl1 = plt.axhline(accs[1], color=c1, alpha=a)
    hl2 = plt.axhline(precs[1], color=c2, alpha=a)
    hl3 = plt.axhline(recs[1], color=c3, alpha=a)
    hl4 = plt.axhline(f1s[1], color=c4, alpha=a)

    # add some text for labels, title and axes ticks
    plt.xlabel("cv fold")
    plt.ylabel("score [%]")
    plt.yticks(np.linspace(0, 100, 21))
    plt.xticks(ind+2*width, [str(i+1) for i in ind])
    plt.ylim([50, 100])
    plt.xlim([-width, folds])

    # reorder the labels
    plt.legend([bar1, hl1, bar2, hl2, bar3, hl3, bar4, hl4], ["accuracy", "mean accuracy", "precision",
                                                              "mean precision", "recall", "mean recall", "f1-score",
                                                              "mean f1-score"], fontsize=8, ncol=4, loc='upper center')

    plt.tight_layout()
    plt.savefig(os.path.join(output, 'cv-results.pdf'), bbox_inches='tight')
    plt.show()
    plt.close('all')


# ______________________________________________________________________________________________________________________
def plot_spectrum(spectrum, rec, cmd_args, frequency_vector):
    f_size = 17
    plt.clf()
    plt.figure(figsize=(20, 12.5))

    color_map = plt.get_cmap('nipy_spectral')
    colors = [color_map(i) for i in np.linspace(0, 1, len(rec.signal_names))]

    # /media/lukas/0084DB135AE0B341/normal_abnormal/normal/tuh_eeg/v0.6.0/edf/004/00000692/s01_2010_10_27/a_1.edf
    class_name = rec.name.split('/')[-8]
    edf_file_name = '/'.join(rec.name.split('/')[-4:])
    plt.title(class_name + ', ' + edf_file_name + ', d=' + str(int(rec.duration)) + 's, ws=' + str(cmd_args.windowsize)
              + 's', fontsize=f_size)

    for electrode_id, electrode in enumerate(spectrum):
        plt.loglog(frequency_vector, electrode, linewidth=2, label=rec.signal_names[electrode_id], alpha=.8,
                   color=colors[electrode_id])

    plt.grid(True)

    plt.xlim([frequency_vector[1]-0.1, rec.sampling_freq/2 + 15])

    plt.xlabel("frequency (Hz)", fontsize=f_size)
    plt.ylabel("median amplitude (mV)", fontsize=f_size)

    x_tick_locations = [.5, 1, 4, 8, 12, 30, 60, 120]
    # since some recordings have higher sampling frequency, adapt plot
    if rec.sampling_freq > 256:
        x_tick_locations += [180]
    x_tick_labels = [str(x_tick_location) for x_tick_location in x_tick_locations]
    plt.xticks(x_tick_locations, x_tick_labels, fontsize=f_size)
    plt.yticks(fontsize=f_size)

    plt.legend(loc="best", ncol=10, mode="expand", fontsize=f_size)

    if cmd_args.output is not None:
        path = os.path.join(cmd_args.output, class_name, edf_file_name.replace('edf', 'pdf').replace('/', '_'))
        logging.info("\t\t\t\tSaving spectrum plot to {}.".format(path))
        plt.savefig(path, bbox_inches="tight")
    else:
        plt.show()
    plt.close("all")


# ______________________________________________________________________________________________________________________
def scatter_features(features_ab, threshold, out_dir=None):
    colors = ['r', 'b']
    for i, features in enumerate(features_ab):
        mean = np.mean(features)
        median = np.median(features)

        plot_rng = RandomState(349834)
        plt.scatter(i + plot_rng.randn(len(features)) * 0.1, features, alpha=0.7, color=colors[i], edgecolor='k')

        plt.plot(-.5 + 2 * i, mean, marker='s', markersize=15, color=colors[i], label='mean')
        plt.plot(-.75 + 2.5 * i, median, marker='o', markersize=15, color=colors[i], label='median')

        yerr_down = abs(mean - np.percentile(features, 25))
        yerr_up = abs(np.percentile(features, 75) - mean)
        plt.errorbar(-.5 + 2 * i, mean, [[yerr_down], [yerr_up]], color='k')
        # print(yerr_down, yerr_up)

        yerr_down = abs(median - np.percentile(features, 25))
        yerr_up = abs(np.percentile(features, 75) - median)
        plt.errorbar(-.75 + 2.5 * i, median, [[yerr_down], [yerr_up]], color='k')
        # print(yerr_down, yerr_up)

    plt.xticks([-.75, -.5, 0, 1, 1.5, 1.75], ['median', 'mean', 'excess_beta', 'normal_eeg', 'mean', 'median'],
               rotation=30)
    plt.yticks(np.arange(0, .0011, 0.0001))

    plt.xlim(-1, 2)
    plt.ylim(-.0001, .001)

    plt.title('median beta amplitude over all channels')
    plt.grid(True)
    # plt.legend(loc='best')

    plt.axhline(threshold, color='m')
    plt.text(.35, threshold*1.1, 'threshold', color='m')

    if out_dir is not None:
        plt.savefig(os.path.join(out_dir, 'scatter.pdf'), bbox_inches="tight")
    else:
        plt.show()
    plt.close("all")


# ______________________________________________________________________________________________________________________
def plot_important_features(out_dir, sorted_zip, title):
    sorted_features, sorted_values = [], []
    for entry in sorted_zip:
        sorted_features.append(entry[0])
        sorted_values.append(entry[1])

    plt.bar(np.arange(len(sorted_features))+.1, sorted_values, width=.8, color='grey')
    plt.title(' '.join(title))

    plt.xticks(np.arange(len(sorted_features))+.5, sorted_features, rotation='vertical', fontsize=8)
    plt.xlim(0, len(sorted_features))
    plt.ylabel("importance [%]")
    plt.xlabel("feature")
    plt.grid(False)
    plt.tight_layout()

    plt.savefig(os.path.join(out_dir, '_'.join(title) + '.pdf'))

    plt.show()
    plt.close('all')


# ______________________________________________________________________________________________________________________
def plot_feature_importances(importances, out_dir, feature_labels=None, n_most_important=10):
    """ plots the 10 most important features per fold.
    alsp plots the mean of all "10 most important features per fold" over all folds.
    :param importances: holds the importance values of all features for every fold
    :param out_dir: output directory
    :param feature_labels: the names of the features in the order they were computed and put into the feature vector
    (which should be the same order as the feature importances. check. double check!)
    :param n_most_important: the number that specifies the most important features that will be plot
    """

    important_features = []
    # importances has shape n_folds x n_features
    # iterate all folds and plot the n most important features
    for fi_id, fi in enumerate(importances):
        n_features = len(fi)
        # create ids for the feature labels
        ids = np.arange(n_features)
        zipped = zip(fi, ids)
        # sort the feature importances with their ids by highest value
        sorted_zipped = sorted(zipped, reverse=True)
        sorted_fimp, sorted_ids = zip(*sorted_zipped)
        # take the n highest values
        fimp, fimp_ids = list(sorted_fimp[:n_most_important]), list(sorted_ids[:n_most_important])

        important_features.extend(fimp_ids)

        # TODO: also add reordered versions of this plot for a better analysis
        plt.bar(np.arange(len(fimp_ids))+0.1, fimp, width=0.8, label="fold " + str(fi_id), color='grey')
        plt.title("10 most important features in fold " + str(fi_id+1))

        if feature_labels is not None:
            fimp_ids = feature_labels[fimp_ids]

        plt.xticks(np.arange(len(fimp_ids))+.5, fimp_ids, rotation='vertical')
        plt.yticks(np.linspace(0, np.max(fimp), 20))
        plt.ylabel("importance [%]")
        plt.xlabel("feature")
        plt.grid(False)
        plt.tight_layout()

        plt.savefig(os.path.join(out_dir, "top_importances_fold" + str(fi_id + 1) + ".pdf"))
        plt.clf()

    # mean of all important features over folds
    important_features = np.unique(important_features)

    # mean over folds
    importances_mean = np.mean(importances, axis=0)
    important_features_mean = importances_mean[important_features]

    if feature_labels is not None:
        important_features = feature_labels[important_features]

    # sort the important features and their ids by highest value
    sorted_zip = sorted(zip(important_features, important_features_mean), key=lambda x: -x[1])
    title = ['mean', 'importance', 'of', 'important', 'features', 'over', 'all', 'folds', 'by', 'value']
    plot_important_features(out_dir, sorted_zip, title)

    # sort the important features and their ids by electrode
    zipped = zip(important_features, important_features_mean)
    sorted_zip = sorted(zipped, key=lambda x: str(x[0]).split('_')[-1] if '_' in str(x[0]) else x[0])
    title = ['mean', 'importance', 'of', 'important', 'features', 'over', 'all', 'folds', 'by', 'electrode']
    plot_important_features(out_dir, sorted_zip, title)

    # sort the important features and their ids by domain
    zipped = zip(important_features, important_features_mean)
    sorted_zip = sorted(zipped, key=lambda x: str(x[0]).split('_')[0] if '_' in str(x[0]) else x[0])
    title = ['mean', 'importance', 'of', 'important', 'features', 'over', 'all', 'folds', 'by', 'domain']
    plot_important_features(out_dir, sorted_zip, title)

    # sort the important features and their ids by frequency
    zipped = zip(important_features, important_features_mean)
    # how to "sort" sex, age and time features that don't have frequencies in their name? make a dummy that puts em to
    # the front
    dummy = [None, 100, None, 100, None]
    sorted_zi = sorted(zipped, key=lambda x: my_io.natural_key(str(x[0]).split('_')[-2]) if '_' in str(x[0]) else dummy)
    title = ['mean', 'importance', 'of', 'important', 'features', 'over', 'all', 'folds', 'by', 'frequency']
    plot_important_features(out_dir, sorted_zi, title)


# ______________________________________________________________________________________________________________________
def plot_feature_importances_spatial(importances, out_dir, feature_labels, n_most_important=5):
    """ plot the feature importances on a scheme of the head to add spatial relationship.
    :param importances: the feature importances as returned by the RF. shape n_folds x n_features
    :param out_dir: the output directory where the plots are saved to
    :param feature_labels: the list of features
    :param n_most_important: the number of features that should be analyzed
    """
    electrode_names = sorted(['A1', 'A2', 'C3', 'C4', 'CZ', 'F3', 'F4', 'F7', 'F8', 'FP1', 'FP2', 'FZ', 'O1', 'O2',
                              'P3', 'P4', 'PZ', 'T3', 'T4', 'T5', 'T6'])

    electrode_to_feature_labels = {
        'A1': [],
        'A2': [],
        'C3': [],
        'C4': [],
        'CZ': [],
        'F3': [],
        'F4': [],
        'F7': [],
        'F8': [],
        'FP1': [],
        'FP2': [],
        'FZ': [],
        'O1': [],
        'O2': [],
        'P3': [],
        'P4': [],
        'PZ': [],
        'T3': [],
        'T4': [],
        'T5': [],
        'T6': [],
    }

    electrode_locations = {
        'PZ': [50, 30],
        'CZ': [50, 50],
        'FZ': [50, 70],

        'O1': [40, 10],
        'O2': [60, 10],

        'T5': [15, 25],
        'T6': [85, 25],

        'P3': [32, 30],
        'P4': [68, 30],

        'F3': [32, 70],
        'F4': [68, 70],

        'F7': [15, 75],
        'F8': [85, 75],

        'FP1': [40, 90],
        'FP2': [60, 90],

        'A1': [-7, 52],
        'T3': [13, 50],
        'C3': [30, 50],
        'C4': [70, 50],
        'T4': [87, 50],
        'A2': [107, 52],
    }

    mean_importances = np.mean(importances, axis=0)

    # assign the mean importances to the respective electrode
    for feature_label_id, feature_label in enumerate(feature_labels):
        for electrode_name in electrode_names:
            if electrode_name == feature_label.split('_')[-1]:
                electrode_to_feature_labels[electrode_name].append(feature_label_id)

    # this version of the plot is not quite as useful as the other version
    # print(electrode_to_feature_labels['O1'], len(electrode_to_feature_labels['O1']))

    # # TODO: make this dynamic for more then 3 most important
    # bars1, labels = [], []
    # bars2, bars3 = [], []
    # # for every electrode sort its assigned feature importances by highest value
    # for electrode_id, electrode in enumerate(electrode_names):
    #     zipped = zip(mean_importances[np.asarray(electrode_to_feature_labels[electrode])],
    #                  electrode_to_feature_labels[electrode])
    #     sorted_zipped = sorted(zipped, reverse=True)
    #     sorted_mean_imp, sorted_mean_imp_ids = zip(*sorted_zipped)
    #     most_important_mean_imp = sorted_mean_imp[:n_most_important]
    #     most_important_mean_imp_ids = sorted_mean_imp_ids[:n_most_important]
    #
    #     print(most_important_mean_imp, feature_labels[np.asarray(most_important_mean_imp_ids)])
    #
    #     # plt.subplot(len(electrode_names)/3, (electrode_id % 3)+1, (electrode_id % 7)+1)
    #
    #     # for i in range(n_most_important):
    #     #     print(i + 1 + electrode_id * n_most_important)
    #     #     plt.bar(i + 1 + electrode_id * n_most_important, most_imported_mean_imp[i], color=colors[electrode_id],
    #     #             alpha=.2*i)
    #
    #     bars1.append(most_important_mean_imp[0])
    #     bars2.append(most_important_mean_imp[1])
    #     bars3.append(most_important_mean_imp[2])
    #     # plt.xlabel(feature_labels[np.asarray(most_imported_mean_imp_ids)])
    #     sub_labels = feature_labels[np.asarray(most_important_mean_imp_ids)]
    #     # sub_labels = [sub_label + ' ' + str(most_imported_mean_imp[i]) for i, sub_label in enumerate(sub_labels)]
    #     labels.extend(sub_labels)
    #
    # # print(len(bars1))
    # # print(np.linspace(0,3*len(bars1)-1, len(bars1)))
    # plt.bar(np.linspace(0, 3*len(bars1)-3, len(bars1)), bars1, color='red')
    # plt.bar(np.linspace(0, 3*len(bars2)-3, len(bars2)) + 1, bars2, color='orangered', alpha=.7)
    # plt.bar(np.linspace(0, 3*len(bars3)-3, len(bars3)) + 2, bars3, color='orange', alpha=.4)
    #
    # for a, b in zip(np.linspace(0, 3*len(bars1)-3, len(bars2)), bars1):
    #     plt.text(a + .17, .05, str(b)[:4], rotation=90, fontsize=6, fontweight='bold')
    #
    # for a, b in zip(np.linspace(0, 3*len(bars2)-3, len(bars2)) + 1, bars2):
    #     plt.text(a + .17, .05, str(b)[:4], rotation=90, fontsize=6, fontweight='bold')
    #
    # for a, b in zip(np.linspace(0, 3*len(bars3)-3, len(bars3)) + 2, bars3):
    #     plt.text(a + .17, .05, str(b)[:4], rotation=90, fontsize=6, fontweight='bold')
    #
    #
    # # TODO: color the labels in the same way as the bars
    # plt.xticks(np.arange(len(labels)), labels, rotation=90, fontsize=8)
    # # plt.xlabel(labels, rottaion=90)
    # plt.xlim([0, 3*len(bars1)])
    # plt.tight_layout()
    # plt.savefig(os.path.join(out_dir, '3_most_important_features_per_electrode.pdf'))
    # plt.show()
    # plt.close('all')

    # _______________________________________________________________________________
    # offset due to aligning strings at the bottom left corner
    offset_x = 6.5
    offset_y = -4
    # the head
    ax = plt.gca()
    c1 = plt.Circle((50+offset_x, 50+offset_y), 50, fill=False, color='black', alpha=.5)
    ax.add_artist(c1)
    c2 = plt.Circle((50+offset_x, 50+offset_y), 40, fill=False, color='black', alpha=.5)
    ax.add_artist(c2)
    # the nose
    plt.plot([47+offset_x, 50+offset_x, 53+offset_x], [100+offset_y, 103+offset_y, 100+offset_y], color='black',
             alpha=.5, linewidth=.1)
    # from ear to ear
    plt.plot([0+offset_x, 100+offset_x], [50+offset_y, 50+offset_y], color='black', alpha=.5, linewidth=.1)
    # from nasion to inion
    plt.plot([50+offset_x, 50+offset_x], [0+offset_y, 100+offset_y], color='black', alpha=.5, linewidth=.1)

    # if n_most_important == 3:
    #     colors = ['red', 'orangered', 'orange']
    # else:
    #     color_map = plt.get_cmap('nipy_spectral')
    #     colors = [color_map(i) for i in np.linspace(0, 1, n_most_important)]

    # put the n most important feature of a electrode at the respective position
    # if only patient features are used, this will crash since the electrode_to_feature_labels map will not contain
    # any entry. this is because "age" and "sex" feature_label does not contain any electrode
    for electrode in electrode_names:
        if not electrode_to_feature_labels[electrode]:
            continue
        zipped = zip(mean_importances[np.asarray(electrode_to_feature_labels[electrode])],
                     electrode_to_feature_labels[electrode])
        sorted_zipped = sorted(zipped, reverse=True)
        sorted_mean_imp, sorted_mean_imp_ids = zip(*sorted_zipped)
        most_important_mean_imp = sorted_mean_imp[:n_most_important]
        most_important_mean_imp_ids = sorted_mean_imp_ids[:n_most_important]

        [x, y] = electrode_locations[electrode]
        for i in range(n_most_important):
            name = '_'.join(feature_labels[most_important_mean_imp_ids[i]].split('_')[1:-1])
            value = str(most_important_mean_imp[i])[:5]
            # plot the electrode in the background
            plt.text(x, y - 10, electrode, fontsize=40, color='blue', alpha=.03)
            if n_most_important == 3:
                # plot the n most important features together with their importances at their location
                plt.text(x, y-i*5.5, name, fontsize=9, fontweight='bold', color='black', alpha=1-.3*i)
                plt.text(x, y-(i+.53)*5.5, str(value),  fontsize=8, color='black', alpha=1-.3*i)
            else:
                plt.text(x, y-1-i*2.5, name + ' ' + value, fontsize=7, fontweight='bold', color='black', alpha=1-.15*i)

    title = [str(n_most_important), 'most', 'important', 'features', 'per', 'electrode', 'spatial']
    plt.title(' '.join(title))
    plt.xlim([-10, 125])
    plt.xticks([])
    plt.ylim([-10, 105])
    plt.yticks([])
    plt.tight_layout()

    plt.show()
    plt.savefig(os.path.join(out_dir, '_'.join(title) + '.pdf'))
    plt.savefig(os.path.join(out_dir, '_'.join(title) + '.png'))
    plt.close('all')

