import torch
import torch.utils.data as data
from mappers import TokenMapper, TokenMapperUnkCategory, UNK, BEGIN, END


class WindowDataset(data.Dataset):
    """
    Pytorch's Dataset derived class to create data sample from
    a path to a file
    """
    def __init__(self, filepath: str, mapper: TokenMapper, window_size: int = 2):
        super().__init__()
        self.filepath = filepath
        self.mapper = mapper
        self.window_size = window_size
        self.samples = []
        self.labels = []

    def _load_file(self) -> None:
        curr_sent = []
        curr_labels = []
        with open(self.filepath, "r", encoding="utf8") as f:
            for line in f:

                if line.startswith("#"):  # comment rows
                    continue

                if line == "\n":  # marks end of sentence

                    self._create_window_samples_from_sent(curr_sent, curr_labels)
                    # clear before reading next sentence
                    curr_sent = []
                    curr_labels = []

                else:  # line for word in a sentence
                    tokens = line.split("\t")
                    word, label = tokens[1], tokens[3]
                    curr_sent.append(word)
                    curr_labels.append(label)

    def _create_window_samples_from_sent(self, sent, labels) -> None:
        sentence_length = len(sent)
        sent = [BEGIN] * self.window_size + sent + [END] * self.window_size

        # iterate for sentence_length times
        # start with index of real word
        for i in range(self.window_size, sentence_length + self.window_size):
            current_sample = sent[i - self.window_size:i + 1 + self.window_size]
            self.samples.append(current_sample)

        for label in labels:
            self.labels.append(label)

    def _get_word_index(self, word) -> int:
        # usual case - word appears in mapping dictionary (seen in train)
        word_to_idx = self.mapper.token_to_idx
        if word in word_to_idx:
            return word_to_idx[word]

        # if word doesn't appear - assign the index of unknown
        return word_to_idx[UNK]

    def _get_label_index(self, label) -> int:
        label_to_idx = self.mapper.label_to_idx
        return label_to_idx[label]

    def __len__(self) -> int:
        # perform lazy evaluation of data loading
        if len(self.samples) == 0:
            self._load_file()

        return len(self.samples)

    def __getitem__(self, item_idx: int) -> tuple:
        # lazy evaluation of data loading
        if len(self.samples) == 0:
            self._load_file()

        # retrieve sample and transform from tokens to indices
        sample, label = self.samples[item_idx], self.labels[item_idx]
        sample_indices = [self._get_word_index(word) for word in sample]
        label_index = self._get_label_index(label)

        # create tensors
        x = torch.tensor(sample_indices)
        y = torch.tensor(label_index)
        return x, y


class WindowDatasetUnkCategories(WindowDataset):
    def __init__(self, filepath: str, mapper: TokenMapperUnkCategory, window_size: int = 2):
        super().__init__(filepath, mapper, window_size)
        self.mapper = mapper

    def _get_word_index(self, word) -> int:
        # usual case - word appears in mapping dictionary (seen in train)
        word_to_idx = self.mapper.token_to_idx
        if word in word_to_idx:
            return word_to_idx[word]

        # if the word doesn't appear - try to find a "smart" unknown pattern
        unknown_categories: dict = self.mapper.unk_categories
        for category, cond_func in unknown_categories.items():
            if cond_func(word):
                return word_to_idx[category]

        # cannot find a smart unknown pattern - return index of general unknown
        return word_to_idx[UNK]
