import numpy as np


class WordSimilarities(object):
    def __init__(self, vocab_path: str, word_vectors_path: str):
        self.vocab = {}
        self.word_vectors: np.array = None

        # init vocab and word_vectors members
        self._load_vocab_file(vocab_path)
        self._load_word_vectors(word_vectors_path)

    @staticmethod
    def vec_similarity(u, v) -> float:
        u_t = u.T
        v_t = v.T
        return u_t @ v / (np.sqrt(u_t @ u) * np.sqrt(v_t @ v))

    def _load_vocab_file(self, vocab_path: str) -> None:
        with open(vocab_path, "r", encoding="utf8") as f:
            for index, word in enumerate(f):
                word = word[:-1]  # remove end of line token
                self.vocab[word] = index

    def _load_word_vectors(self, word_vectors_path: str) -> None:
        self.word_vectors = np.loadtxt(word_vectors_path)

    def get_to_k_similar_words(self, query_word: str, k: int) -> list:
        words_and_vectors = [(word, self.word_vectors[index]) for word, index in self.vocab.items() if word != query_word]
        query_word_vector = self.word_vectors[self.vocab[query_word]]
        words_and_similarities = [(word, self.vec_similarity(vector, query_word_vector)) for word, vector in words_and_vectors]

        # sort the word by their similarity score - greatest similarity first
        sorted_similarities = sorted(words_and_similarities, key=lambda x: x[1], reverse=True)
        # return k most similar
        top_k_similarities = sorted_similarities[:k]

        return top_k_similarities


if __name__ == '__main__':
    vocab = "vocab.txt"
    word_vectors = "wordVectors.txt"
    wordSim = WordSimilarities(vocab, word_vectors)

    # words to query
    k_ = 5
    query_words = ["dog", "england", "john", "explode", "office"]
    word_distance_format = "{0}) {1}: {2:.3f}"
    for word_ in query_words:
        top_words = wordSim.get_to_k_similar_words(word_, k_)

        print(f"Top 5 similar words for the word \"{word_}\" are:")
        for i in range(k_):
            print(word_distance_format.format(i+1, top_words[i][0], top_words[i][1]))
