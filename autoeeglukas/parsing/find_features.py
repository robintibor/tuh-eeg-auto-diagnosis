#!/usr/bin/env python3.5
import argparse
import inspect
import sys
import re
import os

cmd_folder = os.path.realpath(os.path.abspath(os.path.split(inspect.getfile(inspect.currentframe()))[0]))
cmd_folder = os.path.realpath(os.path.join(cmd_folder, ".."))
if cmd_folder not in sys.path:
    sys.path.insert(0, cmd_folder)

from utils import my_io

FEATURE_PATTERNS = {
    "(disrupted (\w+\s)*alpha rhythm)": 0,
    "(slowing (\w+\s)+alpha rhythm)": 0,
    "alpha rhythm": 0,
    "((7.?[5-9]?|[8-9].?[0-9]?|1[0-2].?[0-5]?)[ -]+?hz alpha rhythm)": 0,
    "FIRDA": 0,
    "OIRDA": 0,
    "TIRDA": 0,
    "background slowing": 0,
    "sharp wave": 0,
    "spikes": 0,
    " normal eeg": 0,
    "abnormal eeg": 0,
    "epilep": 0,
    "no epilep": 0,
    "stroke": 0,
    "concus": 0,
    "excess\w* beta": 0,
    "no\w* excess\w* beta": 0,
    "excess\w* theta": 0,
    "generous beta": 0,
    "POSTS": 0,
    "spindles": 0,
    "SREDA": 0,
    "posterior dominant rhythm": 0,
    "seizures": 0,
    "status epilepticus": 0,
    "slowing": 0,
    "artifact": 0,
    "coma": 0,
    "burst": 0,
    "burst suppression": 0,
    "if epilep": 0,
}


def main(cmd_args):
    # support multiple repositories for seizure data set
    if not os.path.isfile(cmd_args.input):
        inputs = my_io.read_all_file_names(cmd_args.input, extension='csv', key='natural')
    else:
        inputs = [cmd_args.input]

    for csv in inputs:
        reports_file = open(csv, 'r')
        reports = reports_file.readlines()
        reports_file.close()

        # standard mode just counts all occurrences of feature patterns
        if not cmd_args.mode:
            for feature_pattern in FEATURE_PATTERNS:
                occurrence = 0
                files, feature_reports = [], []

                for report in reports:
                    m = re.search(feature_pattern, report, re.I)
                    if m is not None:
                        occurrence += 1
                        files.append(report.split(';')[0])
                        feature_reports.append(report)

                FEATURE_PATTERNS[feature_pattern] = occurrence

                # if desired store a index of files containing the feature pattern
                if cmd_args.output is not None:
                    with open(os.path.join(cmd_args.output, feature_pattern + ".txt"), 'w') as text_file:
                        text_file.write('\n'.join(files))

            sorted_by_occurrence = sorted(FEATURE_PATTERNS, key=FEATURE_PATTERNS.get)
            for item in sorted_by_occurrence[::-1]:
                print("Feature '{}' occurs in {:,.2f}% of reports.".format(item, 100 * (FEATURE_PATTERNS[item] /
                                                                                        len(reports))))
            print()

        # interactive mode that reads queries from command line and searches for it in csvs
        else:
            while True:
                query = input("Enter regex that should be searched for:\n")
                if query == "exit" or query == "quit":
                    exit()

                occurrence = 0
                example_report = None
                for report in reports:
                    m = re.search(query, report, re.I)
                    if m is not None:
                        occurrence += 1
                        example_report = report

                print("'{}' occurs {} times. This is {:,.2f}%.".format(query, occurrence, 100*(occurrence/len(reports))))
                if example_report is not None:
                    print(example_report)
                print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('input', type=str, help='input file that contains all reports')
    parser.add_argument('-o', '--output', type=str, help='output directory', default=None, metavar='')

    parser.add_argument('-m', '--mode', action='store_true', help='interactive mode')

    cmd_arguments = parser.parse_args()
    main(cmd_arguments)
