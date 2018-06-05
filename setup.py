
import nltk;
#nltk.download('all');

from nltk.tokenize import word_tokenize;
import string;
from nltk.corpus import stopwords;
from nltk.stem.porter import PorterStemmer;

import numpy as np;

import pandas as pd;
import csv;

NUM_TRAIN = 0.7;
NUM_DEV = 0.15;
NUM_TEST = 0.15;

def text_filter(text):
    '''
    Filter out unncessary factor in a text
    Inputs:
        text: given scopus
    Output:
        cleaned text
    '''
    try:
        tokens = word_tokenize(text);
        # convert to lower case
        tokens = [w.lower() for w in tokens]
        # remove puntuations
        table = str.maketrans('', '', string.punctuation);
        stripped = [w.translate(table) for w in tokens]
        # remove tokens that are not alphbetic
        words = [word for word in stripped if word.isalpha()];
        # filter out stop words
        stop_words  = set(stopwords.words('english'));
        words = [w for w in words if not w in stop_words];
        # stemming of words
        porter = PorterStemmer();
        stemmed = [porter.stem(word) for word in words];
        return ' '.join(w for w in stemmed);
    except TypeError:
        print("TypeError:", text);
        return ' ';

# The whole script begins here
data_spot = pd.read_csv("results/data_spot.txt", sep="\t");
# data_spot now should include all data for trainning + dev + test
# to support downstream tasks, we now apply a filter to eliminated
# unimportant factors
data_spot['description'] = data_spot.apply(lambda row: text_filter(row['description']),
        axis=1);
data_spot['reviews'] = data_spot.apply(lambda row: text_filter(row['reviews']),
        axis=1);
print("Dataset size: ", len(data_spot));
mask = data_spot['description'] != ' ';
data_spot = data_spot[mask];
mask = data_spot['reviews'] != ' ';
data_spot = data_spot[mask];
print("Dataset size: ", len(data_spot));

num_train_dev = int(NUM_TRAIN * len(data_spot));
num_dev_test = num_train_dev + int(NUM_DEV * len(data_spot));
datasets = np.split(data_spot, [num_train_dev, num_dev_test]);

print("Dataset size:", len(datasets[0]), len(datasets[1]), len(datasets[2]));

datasets[0].to_csv("results/data_spot_train.txt", 
    sep="\t", index=None);
datasets[1].to_csv("results/data_spot_dev.txt", 
    sep="\t", index=None);
datasets[2].to_csv("results/data_spot_test.txt", 
    sep="\t", index=None);





#data_spot.to_csv("results/data_spot_cleaned.txt", 
#    sep="\t", index=None)







