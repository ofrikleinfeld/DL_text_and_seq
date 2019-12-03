# This file provides code which you may or may not find helpful.
# Use it if you want, or ignore it.
import os
from collections import Counter


def read_data(fname):
    data = []

    with open(fname, "r", encoding="utf8") as f:
        for line in f:
            label, text = line.strip().lower().split("\t", 1)
            data.append((label, text))

    return data


def text_to_bigrams(text):
    return ["%s%s" % (c1, c2) for c1, c2 in zip(text, text[1:])]


def text_to_unigrams(text):
    return [c for c in text]


def create_dataset(filename, feature_function):
    path = os.path.join(DATA_DIR, filename)
    return [(l, feature_function(t)) for l, t in read_data(path)]


DATA_DIR = os.path.join(os.path.curdir, "../", "data")

TRAIN = create_dataset("train", text_to_bigrams)
DEV = create_dataset("dev", text_to_bigrams)
TEST = create_dataset("test", text_to_bigrams)

# TRAIN = create_dataset("train", text_to_unigrams)
# DEV = create_dataset("dev", text_to_unigrams)

fc = Counter()
for l, feats in TRAIN:
    fc.update(feats)

# 600 most common bigrams in the training set.
vocab = set([x for x, c in fc.most_common(600)])

# label strings to IDs
L2I = {l: i for i, l in enumerate(list(sorted(set([l for l, t in TRAIN]))))}
# feature strings (bigrams) to IDs
F2I = {f: i for i, f in enumerate(list(sorted(vocab)))}