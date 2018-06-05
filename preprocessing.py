import pandas as pd
import numpy as np
import csv

'''
This file stores the functions to transform the raw text data into
inputs for NLP
'''

def create_counts_pmi(w, main_words_file, context_file, file_populate, name_c):
  '''
  Generate count matrix and PMI

  Inputs:
    w = integer
    main_words_file = string (file, one word per line)
    context_file = string (file, one word per line)
    file_populate = string (text of a dataframe)
    name_c = string (name of the column with information)
  Outputs:
    df = dataframe
    pmi = dataframe
  '''
  data_v = pd.read_csv(main_words_file, header = None, names = ["word"])
  data_vc = pd.read_csv(context_file, sep="\n",
      quoting=csv.QUOTE_NONE, header = None, names = ["word"])

  data_populate = create_data_populate(file_populate, name_c)

  dict_counts = improved_neighbors(w, data_v["word"], data_v["word"],
      data_populate)

  df = dic_count_to(dict_counts)

  pmi = get_PMI(df)

  return df, pmi


def get_PMI(df, file_matrix_pmi = None):
    '''
    Generates the PMI matrix from a matrix where the columns are
    the context words and the rows are words. Saves the matrix.

    Input:
        df = dataframe (columns are context words, rows are words)
    Output:
        pmi_matrix = dataframe (columns are context words, rows are words)
    '''
    # Remember that the PMI is log2(p(x,y) / (p(x)*p(y)))

    # General parameters
    all_ocurrences = df.sum().sum()

    # Sum over rows (context words)
    sum_over_rows = df.sum()
    p_j = sum_over_rows / all_ocurrences

    # Sum over columns (words)
    sum_over_cols = df.sum(axis=1)
    p_i = sum_over_cols / all_ocurrences

    # Probability intersection
    p_ij = df / all_ocurrences

    pmi_matrix = p_ij.div(p_j, axis="columns")
    pmi_matrix = pmi_matrix.div(p_i, axis="index")
    pmi_matrix = pmi_matrix.apply(np.log2)

    pmi_matrix[pmi_matrix <= 0] = 0

    if file_matrix_pmi:
        pmi_matrix.to_csv(file_matrix_pmi)

    return pmi_matrix


def dic_count_to(dic_full, file_matrix_pmi = None, cal_pmi = None):
    '''
    Load dictinory of counts and creates the PMI matrix

    Inputs:
        location_dic = string (file of the counts)
    Outputs:
        df_full = dataframe
    '''

    df_full = pd.DataFrame.from_dict(dic_full, orient = "index")
    df_full.fillna(0, inplace = True)

    if cal_pmi:
        df_full = get_PMI(df_full)

    if cal_pmi and file_matrix_pmi:
        df_full = get_PMI(df_full, file_matrix_pmi)

    return df_full


def improved_neighbors(w, context, words, data_populate, file_matrix = None):
    '''
    Improved version of define_neighbors.
    Populates a df with the correct counts of words.

    Inputs:
        w = integer (order)
        context = list/array of context words
        words = list/array of words
        data_populate = list of list
    Output:
        df = DataFrame
    '''
    context = set(context)
    words = set(words)
    dict_counts = {}
    for sentence in data_populate:
        for i, word in enumerate(sentence):
            if word in words:
                if word not in dict_counts.keys():
                    dict_counts[word] = {}
                for j in range(w):
                    # Above
                    try:
                        if sentence[i+j+1] in context:
                            if sentence[i+j+1] in dict_counts[word].keys():
                                val = dict_counts[word][sentence[i+j+1]]
                                dict_counts[word].update(
                                    {sentence[i+j+1] : val + 1})
                            else:
                                dict_counts[word][sentence[i+j+1]] = 1
                    except:
                        None
                    try:
                        if sentence[i-j-1] in context:
                            if sentence[i-j-1] in dict_counts[word].keys():
                                val = dict_counts[word][sentence[i-j-1]]
                                dict_counts[word].update(
                                    {sentence[i-j-1] : val + 1})
                            else:
                                dict_counts[word][sentence[i-j-1]] = 1
                    except:
                        None

    # Save
    if file_matrix:
        np.save(file_matrix, dict_counts)

    return dict_counts


def get_list_words(text):
    '''
    Creates a list of words from a string.

    Input:
        text = string
    Output:
        rv = list
    '''
    rv = text.lower().strip().split()

    return rv


def create_data_populate(filename, name_col):
    '''
    Creates a list of lists, where each list is a list
    of words.

    Input:
        filename = string
        name_col = string
    Output:
        compilation_text = list
    '''
    data = pd.read_csv(filename, sep = "\t")

    important = pd.DataFrame(data[name_col])

    important[name_col] = "<s> " + important[name_col] + " </s>"

    important['input'] = important[name_col].apply(get_list_words)

    compilation_text = []
    for index, row in important.iterrows():
        compilation_text.append(row['input'])

    return compilation_text


def read_file_populate(file):
    '''
    Read the file with the sentences.

    Input:
        file = string
    Output:
        data_populate = list of list
    '''
    with open(file, "r") as ins:
        list_ = []
        for line in ins:
            list_.append(("<s> " + line + "</s>").strip().split())

    return list_



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

