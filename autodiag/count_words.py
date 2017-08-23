from collections import Counter
import re
from collections import namedtuple

import nltk
from nltk import regexp_tokenize
import numpy as np


def extract_dict(filetext):
    filetext = filetext.replace('\t', ' ')
    filetext = re.sub(r"(\w+)\.(\w+)", r"\1. \2", filetext)
    filetext = re.sub(r"(\w+):(\w+)", r"\1. \2", filetext)
    lines = filetext.split('\n')
    current_tag = None
    current_text = ''
    tag_to_text = dict()
    for line in lines:
        new_tag_starts = re.match('[A-Z \t]+:', line)
        if new_tag_starts:
            if current_tag is not None:
                tag_to_text[current_tag] = current_text
            current_tag =  new_tag_starts.group(0)[:-1]
            current_text = line[len(current_tag) + 1:]
            current_tag = current_tag.strip()
        else:
            current_text += line
    assert current_tag is not None
    tag_to_text[current_tag] = current_text
    return tag_to_text


def extract_texts(edf_file_names):
    texts =[]
    for edf_file_name in edf_file_names:
        txt_file_name = edf_file_name[:edf_file_name.rfind('_')] +'.txt'
        texts.append(open(txt_file_name, 'r').read())
    return texts


def clean_words(words):
    cleaned_words = []
    for w in words:
        #w = re.sub(r"([a-zA_Z]+)\.([a-zA_Z]+)", r"\1. \2", w)
        #w = re.sub(r"([a-zA_Z]+)\:([a-zA_Z]+)", r"\1: \2", w)
        ws = re.split(r'\s',w)
        for w in ws:
            if w.endswith('.'):
                w = w[:-1]
            if w.endswith(':'):
                w = w[:-1]
            if w.endswith(','):
                w = w[:-1]
            if not re.match("^\s*$", w):
                cleaned_words.append(w.lower())
    return cleaned_words


def compute_counts_and_relative_frequencies(file_names, labels,
                                            correct_mask, label):

    # for split
    #adapted from https://stackoverflow.com/a/22178786/1469195
    # (removed capturing groups)
    pattern = r'''(?x)               # set flag to allow verbose regexps
                  (?:[A-Z]\.)+         # abbreviations, e.g. U.S.A.
                  | \$?\d+(?:\.\d+)?%? # numbers, incl. currency and percentages
                  | \w+(?:[-']\w+)*    # words w/ optional internal hyphens/apostrophe
                  | [+/\-@&*]        # special characters with meanings
                '''
    file_names = np.array(file_names)
    labels = np.array(labels)
    correct_and_wanted = correct_mask & (labels == label)
    correct_texts = extract_texts(
        file_names[np.flatnonzero(correct_and_wanted)])
    correct_dicts = [extract_dict(t) for t in correct_texts]
    correct_imprs = [d['IMPRESSION'] for d in correct_dicts]
    correct_words = regexp_tokenize("\n".join(correct_imprs), pattern)
    correct_words = clean_words(correct_words)
    correct_counter = Counter(correct_words)

    incorrect_and_wanted = (~correct_mask) & (labels == label)
    incorrect_texts = extract_texts(
        file_names[np.flatnonzero(incorrect_and_wanted)])
    incorrect_dicts = [extract_dict(t) for t in incorrect_texts]
    incorrect_imprs = [d['IMPRESSION'] for d in incorrect_dicts]
    incorrect_words = regexp_tokenize("\n".join(incorrect_imprs), pattern)
    incorrect_words = clean_words(incorrect_words)
    incorrect_counter = Counter(incorrect_words)

    freq_ratios = []
    freq_diffs = []

    for key in set(incorrect_counter.keys()) | set(correct_counter.keys()):
        if (incorrect_counter[key] > 1) or (correct_counter[key] > 1):
            incorrect_freq = incorrect_counter[key] / len(incorrect_words)
            correct_freq = correct_counter[key] / len(correct_words)

            rel_freq = max(incorrect_freq, 1.0 / len(incorrect_words)) / (
                max(correct_freq, 1.0 / len(correct_words)))
            freq_ratios.append((key, rel_freq))
            freq_diff = incorrect_freq - correct_freq
            freq_diffs.append((key, freq_diff))

    sorted_freq_ratios = sorted(freq_ratios, key=lambda t: t[1])
    sorted_freq_diffs = sorted(freq_diffs, key=lambda t: t[1])

    result = namedtuple('WordResult', ['correct_counter',
                                       'incorrect_counter',
                                       'freq_ratios',
                                       'freq_diffs',
                                       'correct_imprs',
                                       'incorrect_imprs'], verbose=False)(
        correct_counter=correct_counter,
        incorrect_counter=incorrect_counter,
        freq_ratios=sorted_freq_ratios,
        correct_imprs=correct_imprs,
        freq_diffs=sorted_freq_diffs,
        incorrect_imprs=incorrect_imprs,
    )
    return result

