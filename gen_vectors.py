
import torch;
import torch.nn as nn;
import torchvision;
import numpy as np;
import torch.optim as optim;

import pandas as pd;
import csv;
from nltk.tokenize import word_tokenize;

'''
--------------------------------------------------------------------------------
CONTROLS HERE
--------------------------------------------------------------------------------
'''
# Embedding size is the number of word to build the vector
EMBEDDING_SIZE = 3000;
# The lenght of the vector which depends on the length of the reviews
SIZE_VEC_APPEND = 500;
# If false, the vectors will be created without the information of
# the spot's description
DESCRIPTION = False

'''
--------------------------------------------------------------------------------
END OF CONTROLS
--------------------------------------------------------------------------------
'''

def load_vocab(path, size):
    f = open(path, "r", encoding='utf-8');
    vocabulary = [];
    vocab_mapper = dict();
    i = 0;
    for word in f:
        word = word.rstrip();
        vocabulary.append(word);
        vocab_mapper[word] = i;
        i += 1;
        if i == size:
            break;
    f.close();
    return vocabulary, vocab_mapper;

def vector_len(number, size_v):
    '''
    Creates a random embedding vector for testing.

    Input:
        number = integer
        size_v = integer
    Output:
        np.array (shape = (size_v, ))
    '''
    np.random.seed(1234567)
    vector = np.random.rand(size_v,) / size_v

    return number * vector


def add_vector(vectors, scores, ids, record, vocab, vocab_mapper,
        add_size = False):
    text = [];
    if DESCRIPTION:
        text.extend(word_tokenize(record['description']));
    text.extend(word_tokenize(record['reviews']));
    vector = np.zeros((len(vocab)));
    for word in text:
        if word in vocab_mapper:
            vector[vocab_mapper[word]] += 1;
    if add_size:
        vec_appendable = vector_len(record['len'], SIZE_VEC_APPEND)
        vector = np.insert(vector, 0, vec_appendable)
    # Adds to the list defined in load_dataset
    vectors.append(vector);
    scores.append(record['review_scores_value']);
    ids.append(record['id']);


def load_dataset(data_path, vector_path, vocabs, vocab_mapper,
        add_size = False):
    vectors = [];
    scores = [];
    ids = [];
    dataset = pd.read_csv(data_path, sep='\t');
    if add_size:
        dataset['len'] = dataset['reviews'].str.len()
    dataset.apply(lambda row: add_vector(vectors, scores, ids, row, vocabs,
        vocab_mapper, add_size), axis=1);
    f = open(vector_path, "w");
    for i in range(len(ids)):
        f.write(str(ids[i]) + '\t');
        f.write(str(scores[i]) + '\t');
        for e in range(len(vectors[i])):
            f.write(str(vectors[i][e]) + ' ');
        f.write('\n');
    f.close();

vocabs, vocab_mapper = load_vocab("results/vocabs.txt", EMBEDDING_SIZE);

if DESCRIPTION:
    # Save the dataset of the case without size of the string
    load_dataset("results/data_spot_train.txt", "results/vector_train.txt",
        vocabs, vocab_mapper);
    load_dataset("results/data_spot_dev.txt", "results/vector_dev.txt",
        vocabs, vocab_mapper);
    load_dataset("results/data_spot_test.txt", "results/vector_test.txt",
        vocabs, vocab_mapper);

    # Save the dataset of the case with SIZE of the string
    load_dataset("results/data_spot_train.txt", "results/vector_train_size.txt",
        vocabs, vocab_mapper, add_size = True);
    load_dataset("results/data_spot_dev.txt", "results/vector_dev_size.txt",
        vocabs, vocab_mapper, add_size = True);
    load_dataset("results/data_spot_test.txt", "results/vector_test_size.txt",
        vocabs, vocab_mapper, add_size = True);
else:
    # nd MEANS NO DESCRIPTION
    # Save the dataset of the case without size of the string
    load_dataset("results/data_spot_train.txt", "results/vector_train_nd.txt",
        vocabs, vocab_mapper);
    load_dataset("results/data_spot_dev.txt", "results/vector_dev_nd.txt",
        vocabs, vocab_mapper);
    load_dataset("results/data_spot_test.txt", "results/vector_test_nd.txt",
        vocabs, vocab_mapper);

    # Save the dataset of the case with SIZE of the string
    load_dataset("results/data_spot_train.txt", "results/vector_train_size_nd.txt",
        vocabs, vocab_mapper, add_size = True);
    load_dataset("results/data_spot_dev.txt", "results/vector_dev_size_nd.txt",
        vocabs, vocab_mapper, add_size = True);
    load_dataset("results/data_spot_test.txt", "results/vector_test_size_nd.txt",
        vocabs, vocab_mapper, add_size = True);

#
