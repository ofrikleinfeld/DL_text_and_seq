import re
from collections import OrderedDict

UNK = "UNK"
CHAR_PAD = "*"
BEGIN = "<s>"
END = "</s>"


class BaseMapper(object):
    """
    Class for mapping discrete tokens in a training set
    to indices and back
    """

    def __init__(self, min_frequency: int = 0, split_char="\t"):
        self.min_frequency = min_frequency
        self.split_char = split_char
        self.token_to_idx = {}
        self.idx_to_token = {}
        self.label_to_idx = {}
        self.idx_to_label = {}

    def serialize(self) -> dict:
        return {
            "min_frequency": self.min_frequency,
            "split_char": self.split_char,
            "token_to_idx": self.token_to_idx,
            "label_to_idx": self.label_to_idx,
            "idx_to_token": self.idx_to_token,
            "idx_to_label": self.idx_to_label
        }

    def deserialize(self, serialized_mapper: dict) -> None:
        self.min_frequency = serialized_mapper["min_frequency"]
        self.split_char = serialized_mapper["split_char"]
        self.token_to_idx = serialized_mapper["token_to_idx"]
        self.label_to_idx = serialized_mapper["label_to_idx"]
        self.idx_to_token = serialized_mapper["idx_to_token"]
        self.idx_to_label = serialized_mapper["idx_to_label"]

    def get_tokens_dim(self) -> int:
        return len(self.token_to_idx)

    def get_labels_dim(self) -> int:
        return len(self.label_to_idx)

    def create_mapping(self, filepath: str) -> None:
        raise NotImplementedError("A concrete mapper class needs to implement create_mapping method ")

    def get_token_idx(self, raw_token: str) -> int:
        raise NotImplementedError("A concrete mapper class needs to implement get_token_idx method ")

    def get_label_idx(self, raw_label: str) -> int:
        raise NotImplementedError("A concrete mapper class needs to implement get_label_idx method ")

    def get_token_from_idx(self, index: int) -> str:
        return self.idx_to_token[index]

    def get_label_from_idx(self, index: int) -> str:
        return self.idx_to_label[index]


class TokenMapper(BaseMapper):
    """
    Class for mapping discrete tokens in a training set
    to indices and back
    """
    def __init__(self, min_frequency: int = 0, split_char="\t"):
        super().__init__(min_frequency, split_char)

    def _init_mappings(self) -> None:
        self.token_to_idx[UNK] = 0
        self.idx_to_token[0] = UNK

    def _remove_non_frequent(self, words_frequencies) -> dict:
        # remove word below min_frequency
        words = OrderedDict()
        for word, frequency in words_frequencies.items():
            if frequency >= self.min_frequency:
                words[word] = 0

        return words

    def create_mapping(self, filepath: str) -> None:
        words_frequencies = OrderedDict()
        labels = OrderedDict()

        with open(filepath, "r", encoding="utf8") as f:
            for line in f:

                # skip empty line (end of sentence_
                if line == "\n":
                    continue

                else:
                    line_tokens = line[:-1].split(self.split_char)  # remove end of line
                    word = line_tokens[0]
                    label = line_tokens[1]

                    words_frequencies[word] = words_frequencies.get(word, 0) + 1
                    labels[label] = 0

        # remove word below min_frequency
        words = self._remove_non_frequent(words_frequencies)

        # init mappings with padding and unknown indices
        self._init_mappings()

        # start index will be different if index 0 marked already as padding
        word_start_index = len(self.token_to_idx)
        label_start_index = len(self.label_to_idx)

        # transform token to indices
        for index, word in enumerate(words.keys(), word_start_index):
            self.token_to_idx[word] = index
            self.idx_to_token[index] = word

        for index, label in enumerate(labels.keys(), label_start_index):
            self.label_to_idx[label] = index
            self.idx_to_label[index] = label

    def get_token_idx(self, raw_token: str) -> int:
        # usual case - word appears in mapping dictionary (seen in train)
        if raw_token in self.token_to_idx:
            return self.token_to_idx[raw_token]

        # if word doesn't appear - assign the index of unknown
        return self.token_to_idx[UNK]

    def get_label_idx(self, raw_label: str) -> int:
        return self.label_to_idx[raw_label]


class TokenMapperUnkCategory(TokenMapper):
    def __init__(self, min_frequency: int = 0, split_char="\t"):
        super().__init__(min_frequency, split_char=split_char)
        self.unk_categories = ["twoDigitNum", "fourDigitNum", "containsDigitAndAlpha",
                               "containsDigitAndDash", "containsDigitAndSlash", "containsDigitAndComma",
                               "containsDigitAndPeriod", "otherNum", "allCaps", "capPeriod",
                               "initCap", "lowerCase", "punkMark", "containsNonAlphaNumeric", "%PerCent%"]

    def get_token_idx(self, raw_token: str) -> int:
        # usual case - word appears in mapping dictionary (seen in train)
        if raw_token in self.token_to_idx:
            return self.token_to_idx[raw_token]

        # if the word doesn't appear - try to find a "smart" unknown pattern
        if self.__is_two_digit_num(raw_token):
            category = "twoDigitNum"
        elif self.__is_four_digit_num(raw_token):
            category = "fourDigitNum"
        elif self.__contains_digit_and_alpha(raw_token):
            category = "containsDigitAndAlpha"
        elif self.__contains_digit_and_dash(raw_token):
            category = "containsDigitAndDash"
        elif self.__contains_digit_and_slash(raw_token):
            category = "containsDigitAndSlash"
        elif self.__contains_digit_and_comma(raw_token):
            category = "containsDigitAndComma"
        elif self.__contains_digit_and_period(raw_token):
            category = "containsDigitAndPeriod"
        elif self.__is_other_num(raw_token):
            category = "otherNum"
        elif self.__is_all_caps(raw_token):
            category = "allCaps"
        elif self.__is_caps_period(raw_token):
            category = "capPeriod"
        elif self.__is_init_cap(raw_token):
            category = "initCap"
        elif self.__is_lower_case(raw_token):
            category = "lowerCase"
        elif self.__is_punk_mark(raw_token):
            category = "punkMark"
        elif self.__is_non_alpha_numeric(raw_token):
            category = "containsNonAlphaNumeric"
        elif self.__is_percent(raw_token):
            category = "%PerCent%"
        else:
            # cannot find a smart unknown pattern - return index of general unknown
            category = UNK

        return self.token_to_idx[category]

    def __contains_digit_and_char(self, word: str, ch: str) -> bool:
        return bool(re.search('\d', word)) and ch in word

    def __is_two_digit_num(self, word: str) -> bool:
        return len(word) == 2 and word.isdigit() and word[0] != '0'

    def __is_four_digit_num(self, word: str) -> bool:
        return len(word) == 4 and word.isdigit() and word[0] != '0'

    def __contains_digit_and_alpha(self, word: str) -> bool:
        return bool(re.search('\d', word)) and bool(re.search('[a-zA-Z_]', word))

    def __contains_digit_and_dash(self, word: str) -> bool:
        return self.__contains_digit_and_char(word, '-')

    def __contains_digit_and_slash(self, word: str) -> bool:
        return self.__contains_digit_and_char(word, '/')

    def __contains_digit_and_comma(self, word: str) -> bool:
        return self.__contains_digit_and_char(word, ',')

    def __contains_digit_and_period(self, word: str) -> bool:
        return self.__contains_digit_and_char(word, '.')

    def __is_other_num(self, word: str) -> bool:
        return word.isdigit()

    def __is_all_caps(self, word: str) -> bool:
        return word.isupper()

    def __is_caps_period(self, word: str) -> bool:
        return len(word) == 2 and word[1] == '.' and word[0].isupper()

    def __is_init_cap(self, word: str) -> bool:
        return len(word) > 1 and word[0].isupper()

    def __is_lower_case(self, word: str) -> bool:
        return word.islower()

    def __is_punk_mark(self, word: str) -> bool:
        return word in (",", ".", ";", "?", "!", ":", ";", "-", '&')

    def __is_non_alpha_numeric(self, word: str) -> bool:
        return bool(re.search('\W', word))

    def __is_percent(self, word: str) -> bool:
        return len(word) > 1 and word[0] == '%' and word[1:].isdigit()

    def _init_mappings(self) -> None:
        # init mappings with BEGIN and END symbols
        self.token_to_idx[BEGIN] = 0
        self.idx_to_token[0] = BEGIN
        self.token_to_idx[END] = 1
        self.idx_to_token[1] = END

        # continue with initiating unknown mappings
        self._init_unknown_mappings()

    def _init_unknown_mappings(self) -> None:
        current_index = len(self.token_to_idx)
        for category in self.unk_categories:
            self.token_to_idx[category] = current_index
            self.idx_to_token[current_index] = category
            current_index += 1

        # add UNK as final fallback
        self.token_to_idx[UNK] = current_index
        self.idx_to_token[current_index] = UNK
        current_index += 1


class TokenMapperWithSubWords(TokenMapperUnkCategory):

    def __init__(self, min_frequency: int = 0, split_char="\t"):
        super().__init__(min_frequency, split_char)
        self.UNK_PREFIX = "UnKnownPrefix"
        self.UNK_SUFFIX = "UnknownSuffix"
        self.prefix_to_index = {}
        self.index_to_prefix = {}
        self.suffix_to_index = {}
        self.index_to_suffix = {}

    def serialize(self) -> dict:
        params_dict = super().serialize()
        params_dict["prefix_to_index"] = self.prefix_to_index
        params_dict["suffix_to_index"] = self.suffix_to_index
        params_dict["index_to_prefix"] = self.index_to_suffix
        params_dict["index_to_suffix"] = self.index_to_suffix

        return params_dict

    def deserialize(self, serialized_mapper: dict) -> None:
        super().deserialize(serialized_mapper)
        self.prefix_to_index = serialized_mapper["prefix_to_index"]
        self.suffix_to_index = serialized_mapper["suffix_to_index"]
        self.index_to_prefix = serialized_mapper["index_to_prefix"]
        self.index_to_suffix = serialized_mapper["index_to_suffix"]

    def get_prefix_index(self, prefix: str) -> int:
        if prefix in self.prefix_to_index:
            return self.prefix_to_index[prefix]
        else:
            return self.prefix_to_index[self.UNK_PREFIX]

    def get_suffix_index(self, suffix: str) -> int:
        if suffix in self.suffix_to_index:
            return self.suffix_to_index[suffix]
        else:
            return self.suffix_to_index[self.UNK_SUFFIX]

    def get_prefix_from_index(self, index: int) -> str:
        return self.index_to_prefix[index]

    def get_suffix_from_index(self, index: int) -> str:
        return self.index_to_suffix[index]

    def get_prefix_dim(self) -> int:
        return len(self.prefix_to_index)

    def get_suffix_dim(self) -> int:
        return len(self.suffix_to_index)

    def _init_mappings(self) -> None:
        super()._init_mappings()
        self.prefix_to_index[self.UNK_PREFIX] = 0
        self.prefix_to_index[BEGIN] = 1
        self.prefix_to_index[END] = 2
        self.suffix_to_index[self.UNK_SUFFIX] = 0
        self.suffix_to_index[BEGIN] = 1
        self.suffix_to_index[END] = 2

        self.index_to_prefix[0] = self.UNK_PREFIX
        self.index_to_prefix[1] = BEGIN
        self.index_to_prefix[2] = END
        self.index_to_suffix[0] = self.UNK_SUFFIX
        self.index_to_suffix[1] = BEGIN
        self.index_to_suffix[2] = END

    def create_mapping(self, filepath: str) -> None:
        super().create_mapping(filepath)

        prefixes_frequencies = OrderedDict()
        suffixes_frequencies = OrderedDict()

        with open(filepath, "r", encoding="utf8") as f:
            for line in f:
                # skip empty line (end of sentence_
                if line == "\n":
                    continue

                else:
                    line_tokens = line[:-1].split(self.split_char)  # remove end of line
                    word = line_tokens[0]
                    prefix = word[:3]
                    suffix = word[-3:]

                    prefixes_frequencies[prefix] = prefixes_frequencies.get(prefix, 0) + 1
                    suffixes_frequencies[suffix] = suffixes_frequencies.get(suffix, 0) + 1

        # remove sub units below min_frequency
        prefixes = self._remove_non_frequent(prefixes_frequencies)
        suffixes = self._remove_non_frequent(suffixes_frequencies)
        prefix_start_index = len(self.prefix_to_index)
        suffix_start_index = len(self.suffix_to_index)

        # transform token to indices
        for index, prefix in enumerate(prefixes.keys(), prefix_start_index):
            self.prefix_to_index[prefix] = index
            self.index_to_prefix[index] = prefix

        for index, suffix in enumerate(suffixes.keys(), suffix_start_index):
            self.suffix_to_index[suffix] = index
            self.index_to_suffix[index] = suffix


class RegularLanguageMapper(BaseMapper):

    def __init__(self, min_frequency: int = 0, split_char: str = "\t"):
        super().__init__(min_frequency, split_char)

    def create_mapping(self, filepath: str = None) -> None:
        token_to_idx = {
            CHAR_PAD: 0,
            "1": 1,
            "2": 2,
            "3": 3,
            "4": 4,
            "5": 5,
            "6": 6,
            "7": 7,
            "8": 8,
            "9": 9,
            "a": 10,
            "b": 11,
            "c": 12,
            "d": 13
        }
        label_to_idx = {
            "1": 1,
            "0": 0
        }

        self.token_to_idx = token_to_idx
        self.label_to_idx = label_to_idx
        self.idx_to_token = {value: key for key, value in token_to_idx.items()}
        self.idx_to_label = {value: key for key, value in label_to_idx.items()}

    def get_token_idx(self, raw_token: str) -> int:
        return self.token_to_idx[raw_token]

    def get_label_idx(self, raw_label: str) -> int:
        return self.label_to_idx[raw_label]

    def get_padding_index(self) -> int:
        return self.get_token_idx(CHAR_PAD)
