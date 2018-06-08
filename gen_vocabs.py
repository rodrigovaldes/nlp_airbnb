
import numpy as np;

import pandas as pd;
import csv;

import nltk;
from nltk.tokenize import word_tokenize;

VOCAB_SIZE = 20000;

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

most_common_words = word_count.most_common(3000)
most_common_df = pd.DataFrame.from_records(most_common_words)
most_common_df[1].plot.line()
plt.show()

# word_count.plot()
common_words = word_count.most_common(VOCAB_SIZE);

count = 0;

vocabs = open("results/vocabs.txt", "w", encoding='utf-8');
#vocabs.write('<UNK>,0');
vocabs.write('<UNK>');
vocabs.write('\n');
threshold = 0.1;
i = 0;
for word, freq in common_words:
    try:
        i += 1;
        count += freq;
        if count > threshold * len(review_words):
            print("To cover", int(threshold*100), "% of the data, need:", i, "word(s), count:", count);
            threshold += 0.1;
        #vocabs.write(word + "," + str(freq));
        vocabs.write(word);
        vocabs.write('\n');
    except UnicodeEncodeError:
        continue;

print(count, len(review_words))

vocabs.close();
