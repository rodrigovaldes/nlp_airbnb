
import numpy as np;

import pandas as pd;
import csv;

import nltk;
from nltk.tokenize import word_tokenize;

VOCAB_SIZE = 100;

review_words = ['<UNK>'];

def count_words(text):
    '''
    update word frequency from text
    Inputs:
        text: given text
    Output:
        None
    '''
    review_words.extend(word_tokenize(text));

# The whole script begins here
data_spot = pd.read_csv("results/data_spot_train.txt", sep="\t");
data_spot.apply(lambda row: count_words(row['description']), axis=1);
data_spot.apply(lambda row: count_words(row['reviews']), axis=1);

word_count = nltk.FreqDist(review_words);
common_words = word_count.most_common(VOCAB_SIZE);

vocabs = open("results/vocabs.txt", "w");
vocabs.write('<UNK>');
vocabs.write('\n');
for word, freq in common_words:
    vocabs.write(word);
    vocabs.write('\n');
vocabs.close();






