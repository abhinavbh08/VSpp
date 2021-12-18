import pickle
import json
import config
import nltk
import os
from collections import Counter
from typing import List


class Vocab:
    """Creates the vocabulary and saves to a pickle file so it can be used later and also during runtime"""

    def __init__(self):
        self.wrd2idx = {}
        self.idx2wrd = {}
        self.idx = 0

    def add_word(self, word: str):
        """Adds the word to the dictionary mappings"""
        if word not in self.wrd2idx:
            self.wrd2idx[word] = self.idx
            self.idx2wrd[self.idx] = word
            self.idx += 1

    def __call__(self, word: str):
        """To call the object as a function"""
        if word in self.wrd2idx:
            return self.wrd2idx[word]
        return self.wrd2idx["_UNK"]

    def __len__(self):
        return len(self.wrd2idx)


def read_data(root_path: str, json_name: str) -> List[List[str]]:
    """
    Reads the json file for captions and returns a list of tokenised captions.

    Returns:
        captions_tokenized: List containing each of the caption tokenised.
    """
    with open(os.path.join(root_path, json_name), "r") as file:
        data = json.load(file)
    data = data["images"]
    captions = []
    for index, image_info in enumerate(data):
        if image_info["split"] == "train":
            for sent in image_info["sentences"]:
                captions.append(nltk.tokenize.word_tokenize(str(sent["raw"]).lower()))

    return captions


def make_vocab(root_path: str, json_name: str, threshold) -> Vocab:
    """
    Calls different functions and creates a vocabulary for the dataset.

    Args:
        root_path: The root path of the data.
        json_name: The name of the json file in which the data is stored.

    Returns:
        vocab: vocab class object
    """
    count = Counter()
    captions = read_data(root_path=root_path, json_name=json_name)
    for sent in captions:
        count.update(sent)

    words = [word for word, cnt in count.items() if cnt >= threshold]

    vocab = Vocab()
    vocab.add_word("_PAD")
    vocab.add_word("_UNK")
    for word in words:
        vocab.add_word(word)

    return vocab


def main():

    root_path = config.root_path
    json_name = config.json_name

    vocab = make_vocab(root_path, json_name, threshold=4)
    with open(os.path.join(config.models_folder, "vocab.pkl"), "wb") as f:
        pickle.dump(vocab, f, pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    main()