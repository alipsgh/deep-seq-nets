
import numpy as np
import re
import torch

from torchtext import data
from torchtext.vocab import Vectors
import spacy
import pandas as pd

from utils.logger import get_logger


def get_char_one_hot_encoding(chars):

    benchmark = 'abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:\'"/\\|_@#$%ˆ&*˜`+-=<>()[]{}\n'
    non_alphabet_chars = set(chars) - set(benchmark)

    num_chars = len(chars)

    one_hot_matrix = np.zeros((num_chars, num_chars), dtype=np.float)
    for i, char in enumerate(chars):
        if char in non_alphabet_chars:
            continue
        else:
            one_hot_matrix[i, i] = 1.

    return one_hot_matrix


class Dataset(object):

    def __init__(self, batch_size, seq_len=None):
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.char_level = None
        self.acceptable_chars = None
        self.vocab = None
        self.embeddings = None
        self.train_iterator = None
        self.valid_iterator = None
        self.test_iterator = None
        self.nlp = spacy.load('en_core_web_sm')

    def tokenize(self, sent):
        if self.char_level:
            regex = '[^' + self.acceptable_chars + ']'
            regex = re.compile(regex)
            text = re.sub(regex, ' ', sent)
            text = re.sub(r'\s+', ' ', text)
            output = list(text[::-1])
        else:
            output = [x.text for x in self.nlp.tokenizer(sent) if x.text != " "]
        return output

    def load_data(self, train_file, test_file, valid_file=None, embedding_source=None):
        """
        Loads the data from files
        Sets up iterators for training, validation and test data
        Also create vocabulary and word embeddings based on the data
        :param train_file: absolute path to training file
        :param test_file: absolute path to test file
        :param valid_file: absolute path to validation file
        :param embedding_source: absolute path to file containing word embeddings (GloVe/Word2Vec)
        :return:
        """

        if embedding_source == 'char_one_hot':
            self.char_level = True
            self.acceptable_chars = 'a-z0-9,;.!?:"\'/|_@#$%^&*~`+=<>(){}\n\\[\\]\\\\-'

        # Creating Field for data
        # If the self.seq_len is none, then the length will be flexible.
        text = data.Field(sequential=True, tokenize=self.tokenize, lower=True, fix_length=self.seq_len)
        label = data.Field(sequential=False, use_vocab=False)
        data_fields = [("text", text), ("label", label)]

        # Load data from pd.DataFrame into torchtext.data.Dataset
        train_df = pd.read_csv(train_file)
        train_df = train_df[["text", "label"]]
        train_examples = [data.Example.fromlist(example, data_fields) for example in train_df.values.tolist()]
        train_data = data.Dataset(train_examples, data_fields)

        # If validation file exists, load it. Otherwise get validation data from training data
        if valid_file is not None:
            valid_df = pd.read_csv(valid_file)
            valid_df = valid_df[["text", "label"]]
            valid_examples = [data.Example.fromlist(example, data_fields) for example in valid_df.values.tolist()]
            valid_data = data.Dataset(valid_examples, data_fields)
        else:
            train_data, valid_data = train_data.split(split_ratio=0.9)

        test_df = pd.read_csv(test_file)
        test_df = test_df[["text", "label"]]
        test_examples = [data.Example.fromlist(example, data_fields) for example in test_df.values.tolist()]
        test_data = data.Dataset(test_examples, data_fields)

        if embedding_source is None:
            text.build_vocab(train_data)
        elif embedding_source == 'char_one_hot':
            text.build_vocab(train_data)
            embeddings = get_char_one_hot_encoding(text.vocab.itos)
            text.vocab.set_vectors(text.vocab.stoi, torch.from_numpy(embeddings), len(text.vocab.stoi))
            self.embeddings = text.vocab.vectors
        else:
            text.build_vocab(train_data, vectors=Vectors(embedding_source))
            self.embeddings = text.vocab.vectors

        self.vocab = text.vocab

        self.train_iterator = data.BucketIterator(train_data, batch_size=self.batch_size, sort_key=lambda x: len(x.text), repeat=False, shuffle=True)
        self.valid_iterator = data.BucketIterator(valid_data, batch_size=self.batch_size, sort_key=lambda x: len(x.text), repeat=False, shuffle=False)
        self.test_iterator = data.BucketIterator(test_data, batch_size=self.batch_size, sort_key=lambda x: len(x.text), repeat=False, shuffle=False)

        logger = get_logger()
        logger.info("{} training examples are loaded.".format(len(train_data)))
        logger.info("{} validation examples are loaded.".format(len(valid_data)))
        logger.info("{} test examples are loaded.".format(len(test_data)))

