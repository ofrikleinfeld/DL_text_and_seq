import hashlib
from typing import Dict
from collections import OrderedDict

from pos_and_ner.mappers import BaseMapperWithPadding, TokenMapper

UNK = "unkkkkkkkkkkk"
WORD_PAD = "paddddddddddddd"


class SNLIMapperWithGloveIndices(BaseMapperWithPadding, TokenMapper):

    def __init__(self, min_frequency: int = 5, split_char: str = "\t", glove_words_path: str = ""):
        super().__init__(min_frequency=min_frequency, split_char=split_char)
        self.unknown_label_symbol = "-"
        self.word_to_glove_idx = {}
        self.glove_words_path = glove_words_path

    def serialize(self) -> dict:
        serialization = super().serialize()
        serialization["word_to_glove_idx"] = self.word_to_glove_idx
        serialization["unknown_label_symbol"] = self.unknown_label_symbol
        return serialization

    def deserialize(self, serialized_mapper: dict) -> None:
        super().deserialize(serialized_mapper)
        self.word_to_glove_idx = serialized_mapper["word_to_glove_idx"]
        self.unknown_label_symbol = serialized_mapper["unknown_label_symbol"]

    def create_mapping(self, filepath: str) -> None:
        words_frequencies = OrderedDict()
        labels = OrderedDict()
        glove_words_indices = self._load_glove_words_indices()

        # first phase - extract all words from training set
        with open(filepath, "r", encoding="utf8") as f:
            header_line = True
            for line in f:
                # skip header line, first line in the file
                if header_line:
                    header_line = False
                    continue

                line_tokens = line[:-1].split(self.split_char)  # remove end of line
                label = line_tokens[0]
                # skip samples with unknown label - i.e "-" label
                if label != self.unknown_label_symbol:

                    sentence_1 = [word.lower() for word in line_tokens[5].split()]
                    sentence_2 = [word.lower() for word in line_tokens[6].split()]

                    labels[label] = 0
                    sentences_words = sentence_1 + sentence_2
                    for word in sentences_words:
                        words_frequencies[word] = words_frequencies.get(word, 0) + 1

        # only take into account words the appear in training set enough times
        # and has glove pre trained vector
        # all other words will be treated as OOV

        # remove non frequent
        words = self._remove_non_frequent(words_frequencies)

        # first init unknown tokens and padding token
        self._init_unknown_tokens()
        self._init_padding_idx()

        # start index will be different if index 0 marked already as padding
        word_index = len(self.token_to_idx)
        label_index = len(self.label_to_idx)

        for word in words.keys():
            # only if word has pre trained glove vector
            if word in glove_words_indices:
                self.token_to_idx[word] = word_index
                self.idx_to_token[word_index] = word
                self.word_to_glove_idx[word] = glove_words_indices[word]
                word_index += 1

        for label in labels.keys():
            self.label_to_idx[label] = label_index
            self.idx_to_label[label_index] = label
            label_index += 1

    def _load_glove_words_indices(self) -> Dict[str, int]:
        pre_trained_words_indices = {}
        with open(self.glove_words_path, "r", encoding="utf8") as f:
            for index, word in enumerate(f):
                word = word[:-1]  # remove end of line token
                pre_trained_words_indices[word] = index

        return pre_trained_words_indices

    def _init_unknown_tokens(self) -> None:
        """
        As described in the article
        "OOV words are hashed to one of 100 random embeddings
        so we need 10 different "unknown" tokens
        """
        unk_template = UNK + "_{idx}"
        unk_tokens = {unk_template.format(idx=i): i for i in range(100)}
        self.token_to_idx.update(unk_tokens)
        self.idx_to_token = {value: key for key, value in self.token_to_idx.items()}

    def _init_padding_idx(self) -> None:
        padding_idx = len(self.token_to_idx)
        self.token_to_idx[WORD_PAD] = padding_idx
        self.idx_to_token[padding_idx] = WORD_PAD

    def get_token_idx(self, raw_token: str) -> int:
        lower_raw_token = raw_token.lower()
        if lower_raw_token not in self.token_to_idx:

            # compute hash bucket - one of 100 buckets
            hash_object = hashlib.sha256(lower_raw_token.encode("utf-8"))
            hex_dig = hash_object.hexdigest()
            bucket = int(hex_dig, 16) % 100
            unknown_token = f"{UNK}_{bucket}"
            token = unknown_token
        else:
            token = lower_raw_token

        return self.token_to_idx[token]

    def get_label_idx(self, raw_label: str) -> int:
        return self.label_to_idx[raw_label]

    def get_padding_index(self) -> int:
        return self.get_token_idx(WORD_PAD)

    def get_label_padding_index(self) -> int:
        raise AttributeError("SNLI mapper doesn't have padding symbol because it is a multi class classification problem not sequence")

    def get_padding_symbol(self) -> str:
        return WORD_PAD
