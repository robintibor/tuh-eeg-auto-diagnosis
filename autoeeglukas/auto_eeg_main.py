#!/usr/bin/env python3.5
from datetime import datetime, date
import textwrap as _textwrap
import argparse
import warnings
import logging
import os

from windowing import data_splitter
from cleaning import data_cleaner
from utils import my_io
import pipeline


# ______________________________________________________________________________________________________________________
class LineWrapRawTextHelpFormatter(argparse.RawDescriptionHelpFormatter):
    # this makes the help look much nicer since it allows for a higher line width
    def _split_lines(self, text, width):
        text = self._whitespace_matcher.sub(' ', text).strip()
        return _textwrap.wrap(text, 96)


# ______________________________________________________________________________________________________________________
def main(cmd_args, not_known):
    today, now = date.today(), datetime.time(datetime.now())

    if not cmd_args.no_pre:
        cmd_args.output = os.path.join(cmd_args.output, "autoeeg_" + str(today) + '_' + str(now)[:-7].replace(':', '-'),
                                       '')
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    # create output directories
    my_io.check_out(cmd_args.output, cmd_args.input)
    logname = os.path.join(cmd_args.output, ''.join(cmd_args.output.split('/')[-2:]) + '.log')
    formatter = logging.Formatter("[%(levelname)s] (%(module)s:%(lineno)d) %(message)s")
    logging.basicConfig(level=cmd_args.verbosity)

    rootlogger = logging.getLogger()
    rootlogger.handlers = []

    # add a logger that logs to console
    streamhandler = logging.StreamHandler()
    streamhandler.setFormatter(formatter)
    rootlogger.addHandler(streamhandler)

    # add a logger that logs to file
    filhandler = logging.FileHandler(logname)
    filhandler.setFormatter(formatter)
    rootlogger.addHandler(filhandler)

    logging.info('\tStarted on {} at {}.'.format(today, now))

    if not_known:
        logging.error('\tFollowing parameters could not be interpreted: {}.'.format(not_known))
    logging.info('\tWill use the following parameters: {}.'.format(cmd_args))

    pl = pipeline.Pipeline()
    pl.run(cmd_args)
    
    logging.info('\tFinished on {} at {}.\n\n'.format(date.today(), datetime.time(datetime.now())))

# ______________________________________________________________________________________________________________________
if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=LineWrapRawTextHelpFormatter)

    parser.add_argument('input', type=str, nargs='+', help='list of input directories / feature files if '
                                                           'preprocessing is skipped (-no-pre)')

    parser.add_argument('output', type=str, help='output directory')

    parser.add_argument('-w', '--window', type=str, default='boxcar', choices=data_splitter.WINDOWS, metavar='',
                        help='window function to improve fourier transform. choose between: '
                             + ', '.join(data_splitter.WINDOWS))

    parser.add_argument('-ws', '--windowsize', '--window-size', type=float, default=2, metavar='',
                        help='length of the window in seconds')

    parser.add_argument('--overlap', type=int, help='percentage of overlap of time windows', default=0, metavar='')

    parser.add_argument('--bands', type=str, nargs='+', metavar='', help='frequency bands used in computation',
                        default='1,4,8,12,18,24,30,60,90')

    parser.add_argument('--subset', type=int, default=None, metavar='', help='if a performance estimate on a random '
                                                                             'subset of the data is desired')

    parser.add_argument('--visualize', type=str, nargs='+', metavar='', default=['train', 'pred', 'post'],
                        help='specify what steps of the pipeline should be visualized. choose between: ' +
                             ', '.join(['all', 'pre', 'train', 'pred', 'post', 'none']))

    parser.add_argument('--verbosity', type=int, default=20, metavar='', choices=[0, 10, 20, 30, 40, 50],
                        help='verbosity level of logging: ' + ', '.join(str(s) for s in [0, 10, 20, 30, 40, 50]))

    # TODO: this is basically another mutually exclusive group. add this (needs changing lots of if statements...)
    parser.add_argument('--no-pre', '--nopre', action='store_true',
                        help='skips the preprocessing step. activate this when features are already available.')

    parser.add_argument('--pre-only', '--preonly', action='store_true',
                        help='set this if you want to compute features only. no predictions.')

    # group1 = parser.add_mutually_exclusive_group()
    # group1.add_argument('--no-pre', action='store_false', help='skips the preprocessing step. activate this when '
    #                                                            'features are already available.')
    # group1.add_argument('--pre-only', action='store_true', help='set this if you want to compute features only. no '
    #                                                             'predictions.')
    # group1.set_defaults(pre_only=True)
    # parser.add_argument_group(group1)

    parser.add_argument('--auto-skl', '--autoskl', type=int, nargs=2, metavar='', default=[0, 0],
                        help='toggles the use of auto-sklearn. requires for a total and a run time budget.')

    parser.add_argument('--no-up', '--noup', action='store_true',
                        help='use this option if you do NOT want to upload the results to google spreadsheet.')

    parser.add_argument('--n-proc', '--nproc', type=int, default=1, metavar='',
                        help='number of processes used to read data and compute features')

    parser.add_argument('--domain', type=str, nargs='+', metavar='', default=['all'],
                        choices=['patient', 'fft', 'time', 'pyeeg', 'dwt', 'all'],
                        help='list of features to be computed / used for classification if preprocessing is skipped '
                             '(--no-pre). choose between: ' + ', '.join(['patient', 'fft', 'time', 'pyeeg', 'dwt',
                                                                        'all']))

    parser.add_argument('--feats', type=str, nargs='+', metavar='', default=['all'],
                        help='list of features to be computed / used for classification if preprocessing is '
                             'skipped (--no-pre). take a look at file "features.list" for choices.')

    # TODO: add choices? hard because you can directly specifiy individual electrodes but you can also specify 'o' to
    # jointly select o1 and o2
    parser.add_argument('--elecs', type=str, nargs='+', metavar='', default=['all'],
                        help='list of electrodes to be used for feature computation / prediction of preprocessing is '
                        'skipped (--no-pre). choose between: ' + ', '.join(data_cleaner.ELECTRODES) + ', all.')

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--perrec', dest='perrec', action='store_true', help='computes features per recording')
    group.add_argument('--perwin', dest='perrec', action='store_false', help='computes feature per window')
    group.set_defaults(perrec=True)
    parser.add_argument_group(group)

    cmd_arguments, unknown = parser.parse_known_args()
    main(cmd_arguments, unknown)
