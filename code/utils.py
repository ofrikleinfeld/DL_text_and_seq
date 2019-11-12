# This file provides code which you may or may not find helpful.
# Use it if you want, or ignore it.
import os
from collections import Counter


def read_data(fname):
    data = []
    absolute_fame = os.path.join(DATA_DIR, fname)

    for line in open(absolute_fame, "r", encoding="utf8"):
        label, text = line.strip().lower().split("\t", 1)
        data.append((label, text))
    return data


def text_to_bigrams(text):
    return ["%s%s" % (c1, c2) for c1, c2 in zip(text, text[1:])]


DATA_DIR = os.path.join(os.path.curdir, "../", "data")
TRAIN = [(l, text_to_bigrams(t)) for l, t in read_data("train")]
DEV = [(l, text_to_bigrams(t)) for l, t in read_data("dev")]

fc = Counter()
for l, feats in TRAIN:
    fc.update(feats)

# 600 most common bigrams in the training set.
vocab = set([x for x, c in fc.most_common(600)])

# label strings to IDs
L2I = {l: i for i, l in enumerate(list(sorted(set([l for l, t in TRAIN]))))}
# feature strings (bigrams) to IDs
F2I = {f: i for i, f in enumerate(list(sorted(vocab)))}