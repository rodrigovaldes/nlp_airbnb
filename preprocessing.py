import pandas as pd
import numpy as np

'''
This file stores the functions to transform the raw text data into
inputs for NLP
'''


def plot_histogram(vector, title = None):
    '''
    Plots an histogram

    Input:
        vector = array, series, list
    '''

    vector = vector.dropna()
    mu = np.mean(vector)
    sigma = np.std(vector)

    n, bins, patches = plt.hist(vector, normed=1, facecolor='green', alpha=0.75)
    n, bins, patches = plt.hist(vector, facecolor='green', alpha=0.75)

    # add a 'best fit' line
    y = mlab.normpdf( bins, mu, sigma)
    l = plt.plot(bins, y, 'r--', linewidth=1)

    plt.xlabel('Value')
    plt.ylabel('Probability')
    if title:
        plt.title(title)
    plt.grid(True)

    # plt.show()
    plt.savefig('images/g_'+title)
    plt.close()


def define_subset(data, min_reviews, max_reviews = None,
        name_col = 'number_of_reviews'):
    '''
    Creates a subset of a dataframe

    Inputs:
        data = dataframe
        min_reviews = integer
        name_col = string
    Outpus:
        data = dataframe (filtered dataframe)
        gb = groupby object
    '''
    print("The len of the old data is: ", len(data))

    mask = data[name_col] >= min_reviews

    data = data[mask]

    if max_reviews:
        mask2 = data[name_col] <= max_reviews
        data = data[mask2]

    print("The len of the new data is: ", len(data))

    return data

