import numpy as np

ENCODING = "utf-8"


def parse_glove(glove_filepath: str, vector_output: str, indices_output: str) -> None:
    f_word_indices = open(indices_output, "w", encoding=ENCODING)
    f_vector = open(vector_output, "w", encoding=ENCODING)

    with open(glove_filepath, "r", encoding=ENCODING) as f:
        for line in f:
            line = line[:-1]  # remove end of line
            line_tokens = line.split()
            word = line_tokens[0] + "\n"
            vector = " ".join(line_tokens[1:]) + "\n"
            f_word_indices.write(word)
            f_vector.write(vector)

    # close files
    f_word_indices.close()
    f_vector.close()


if __name__ == '__main__':
    glove_path = "glove.6B.300d.txt"
    output_vectors_path = "glove_vectors.txt"
    output_words_path = "glove_words.txt"
    parse_glove(glove_path, output_vectors_path, output_words_path)
