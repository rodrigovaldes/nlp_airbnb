
import torch;
import torch.nn as nn;
import torchvision;
import numpy as np;
import torch.optim as optim;

import pandas as pd;
import csv;
from nltk.tokenize import word_tokenize;

EMBEDDING_SIZE = 3000;

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

def add_vector(vectors, scores, ids, record, vocab, vocab_mapper):
    text = [];
    text.extend(word_tokenize(record['description']));
    text.extend(word_tokenize(record['reviews']));
    vector = np.zeros((len(vocab)));
    for word in text:
        if word in vocab_mapper:
            vector[vocab_mapper[word]] += 1;
    vectors.append(vector);
    scores.append(record['review_scores_value']);
    ids.append(record['id']);



def load_dataset(data_path, vector_path, vocabs, vocab_mapper):
    vectors = [];
    scores = [];
    ids = [];
    dataset = pd.read_csv(data_path, sep='\t');
    dataset.apply(lambda row: add_vector(vectors, scores, ids, row, vocabs, vocab_mapper),
            axis=1);
    f = open(vector_path, "w");
    for i in range(len(ids)):
        f.write(str(ids[i]) + '\t');
        f.write(str(scores[i]) + '\t');
        for e in range(len(vectors[i])):
            f.write(str(vectors[i][e]) + ' ');
        f.write('\n');
    f.close();

vocabs, vocab_mapper = load_vocab("results/vocabs.txt", EMBEDDING_SIZE);
load_dataset("results/data_spot_train.txt", "results/vector_train.txt", vocabs, vocab_mapper);
load_dataset("results/data_spot_dev.txt", "results/vector_dev.txt", vocabs, vocab_mapper);
load_dataset("results/data_spot_test.txt", "results/vector_test.txt", vocabs, vocab_mapper);




