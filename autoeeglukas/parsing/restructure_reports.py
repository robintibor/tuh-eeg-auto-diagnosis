#!/usr/bin/python3.5
import argparse
import inspect
import csv
import sys
import os

cmd_folder = os.path.realpath(os.path.abspath(os.path.split(inspect.getfile(inspect.currentframe()))[0]))
cmd_folder = os.path.realpath(os.path.join(cmd_folder, ".."))
if cmd_folder not in sys.path:
    sys.path.insert(0, cmd_folder)

from utils import my_io

# most_common_report_items = ["introduction",
#                             "descriptionoftherecord",
#                             "clinicalhistory",
#                             "medications",
#                             "impression",
#                             "clinicalcorrelation",
#                             "hr",
#                             # "abnormaldischarges",
#                             # "reasonforstudy",
#                             # "technicaldifficulties",
#                             # "seizures",
#                             ]


def read_report(path):
    report = None
    f = open(path, encoding="utf-8")
    # some records cannot be read with utf-8 encoding
    try:
        report = f.readlines()
    except UnicodeDecodeError:
        f.close()

        # however, they can be opened with latin-8 encoding
        f = open(path, encoding="latin-1")
        try:
            report = f.readlines()
        except UnicodeDecodeError:
            print("ERROR: Can't read {} with either {} or {} encoding.".format(path, "utf-8", "latin-1"))

            f.close()

    return report


def clear_report(report):
    report = ''.join(report)
    report = report.replace(';', ' ')
    report = report.replace('^M', '\n')
    report = report.replace('\r', '\n')
    report = report.replace('\n', ' ')
    report = report.strip()
    return ' '.join(report.split())


def main(cmd_args):
    absolute_file_paths = my_io.read_all_file_names(cmd_args.input, extension=".txt")

    with open(os.path.join(cmd_args.output, cmd_args.name), "w") as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=';')

        for absolute_file_path in absolute_file_paths:
            report = read_report(absolute_file_path)

            report = clear_report(report)

            relative_file_path = os.path.join(*absolute_file_path.split('/')[4:])
            csv_writer.writerow([relative_file_path] + [report])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('input', type=str, help='root dir of report files')
    parser.add_argument('output', type=str, help='output directory')

    parser.add_argument('-n', '--name', type=str, default='some_reports.csv', metavar='', help='name of output csv')

    cmd_arguments = parser.parse_args()
    main(cmd_arguments)
