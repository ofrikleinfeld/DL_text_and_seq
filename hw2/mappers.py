import re

PADD = "PADD"
UNK = "UNK"


class TokenMapper(object):
    """
    Class for mapping discrete tokens in a training set
    to indices and back
    """
    def __init__(self, min_frequency: int = 0, with_padding: bool = True):
        self.with_padding = with_padding
        self.min_frequency = min_frequency
        self.token_to_idx = {}
        self.idx_to_token = {}
        self.label_to_idx = {}
        self.idx_to_label = {}

    def serialize(self) -> dict:
        return {
            "with_padding": self.with_padding,
            "min_frequency": self.min_frequency,
            "token_to_idx": self.token_to_idx,
            "label_to_idx": self.label_to_idx,
            "idx_to_token": self.idx_to_token,
            "idx_to_label": self.idx_to_label
        }

    @classmethod
    def deserialize(cls, serialized_mapper: dict):
        mapper = cls()

        mapper.with_padding = serialized_mapper["with_padding"]
        mapper.min_frequency = serialized_mapper["min_frequency"]
        mapper.token_to_idx = serialized_mapper["token_to_idx"]
        mapper.label_to_idx = serialized_mapper["label_to_idx"]
        mapper.idx_to_token = serialized_mapper["idx_to_token"]
        mapper.idx_to_label = serialized_mapper["idx_to_label"]

        return mapper

    def get_tokens_dim(self) -> int:
        return len(self.token_to_idx)

    def get_labels_dim(self) -> int:
        return len(self.label_to_idx)

    def _init_mappings(self) -> None:
        if self.with_padding:
            self.token_to_idx[PADD] = 0
            self.token_to_idx[UNK] = 1
            self.idx_to_token[0] = PADD
            self.idx_to_token[1] = UNK

            self.label_to_idx[PADD] = 0
            self.idx_to_label[0] = PADD

        else:
            self.token_to_idx[UNK] = 0
            self.idx_to_token[0] = UNK

    def _remove_non_frequent(self, words_frequencies) -> set:
        # remove word below min_frequency
        words = set()
        for word, frequency in words_frequencies.items():
            if frequency >= self.min_frequency:
                words.add(word)

        return words

    def create_mapping(self, filepath: str) -> None:
        words_frequencies = {}
        labels = set()

        with open(filepath, "r", encoding="utf8") as f:
            for line in f:
                if line != "\n" and not line.startswith("#"):
                    line_tokens = line.split("\t")
                    word = line_tokens[1]
                    label = line_tokens[3]

                    words_frequencies[word] = words_frequencies.get(word, 0) + 1
                    labels.add(label)

        # remove word below min_frequency
        words = self._remove_non_frequent(words_frequencies)

        # init mappings with padding and unknown indices
        self._init_mappings()

        # start index will be different if index 0 marked already as padding
        word_start_index = len(self.token_to_idx)
        label_start_index = len(self.label_to_idx)

        # transform token to indices
        for index, word in enumerate(words, word_start_index):
            self.token_to_idx[word] = index
            self.idx_to_token[index] = word

        for index, label in enumerate(labels, label_start_index):
            self.label_to_idx[label] = index
            self.idx_to_label[index] = label


class TokenMapperUnkCategory(TokenMapper):
    def __init__(self, min_frequency: int = 0):
        super().__init__(min_frequency, with_padding=False)
        self.unk_categories = {
            'twoDigitNum': lambda w: len(w) == 2 and w.isdigit() and w[0] != '0',
            'fourDigitNum': lambda w: len(w) == 4 and w.isdigit() and w[0] != '0',
            'containsDigitAndAlpha': lambda w: bool(re.search('\d', w)) and bool(re.search('[a-zA-Z_]', w)),
            'containsDigitAndDash': lambda w: self._contains_digit_and_char(w, '-'),
            'containsDigitAndSlash': lambda w: self._contains_digit_and_char(w, '/'),
            'containsDigitAndComma': lambda w: self._contains_digit_and_char(w, ','),
            'containsDigitAndPeriod': lambda w: self._contains_digit_and_char(w, '.'),
            'otherNum': lambda w: w.isdigit(),
            'allCaps': lambda w: w.isupper(),
            'capPeriod': lambda w: len(w) == 2 and w[1] == '.' and w[0].isupper(),
            'initCap': lambda w: len(w) > 1 and w[0].isupper(),
            'lowerCase': lambda w: w.islower(),
            'punkMark': lambda w: w in (",", ".", ";", "?", "!", ":", ";", "-", '&'),
            'containsNonAlphaNumeric': lambda w: bool(re.search('\W', w)),
            'percent': lambda w: len(w) > 1 and w[0] == '%' and w[1:].isdigit()
        }

    @staticmethod
    def _contains_digit_and_char(word, ch) -> bool:
        return bool(re.search('\d', word)) and ch in word

    def _init_mappings(self) -> None:
        for index, category in enumerate(self.unk_categories.keys()):
            self.token_to_idx[category] = index
            self.idx_to_token[index] = category

        # add UNK as final fallback
        current_index = len(self.token_to_idx)
        self.token_to_idx[UNK] = current_index
        self.idx_to_token[current_index] = UNK
