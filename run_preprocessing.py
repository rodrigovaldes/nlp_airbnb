import pandas as pd
import numpy as np
import csv 
from preprocessing import *
seed_ = 82848392

'''
This files follows the stepts to transform the data
from Inside AirBnb to inputs for NLP
'''

'''
------------------------------------------------------------------------------
# 0. Load the raw data
------------------------------------------------------------------------------
'''
listings = pd.read_csv("data/listings.csv")
listings.dropna(subset = ['review_scores_rating'], inplace = True)

reviews = pd.read_csv("data/reviews.csv")

'''
------------------------------------------------------------------------------
# 1. UNIT OF ANALYSIS: AirBnb Spot
------------------------------------------------------------------------------
'''

# 1.1 Create Data For the Spot 
# ("sum" of all that the host say about the spot)

# Keep those with 5 - 100 reviews
listings_relevant = define_subset(listings, 5, 100)

text_features = ['summary',
       'space', 'description', 'neighborhood_overview',
       'notes', 'transit', 'access', 'interaction', 'house_rules',
       'host_about']

text_df = listings_relevant[text_features]

all_text = pd.DataFrame(text_df.fillna('').sum(axis=1)).rename(
    columns = {0: "description"})
all_text['id'] = listings_relevant['id']
all_text['review_scores_value'] = listings_relevant['review_scores_value']
all_text['review_scores_cleanliness'] = listings_relevant[
    'review_scores_cleanliness']
# Note: all_text["description"] includes all the text that describe the spot
all_text.dropna(inplace=True)


# 1.2 Create Data For Reviews
#     (sum of all that the reviewers say about a spot)

reviews_relevant = reviews[['listing_id', 'comments']]
reviews_relevant.dropna(inplace=True)
compilation_reviews = pd.DataFrame(reviews_relevant.groupby(["listing_id"]
    ).apply(lambda x: " ".join(x["comments"]))).rename(
    columns = {0: 'reviews'})
compilation_reviews['id'] = compilation_reviews.index
compilation_reviews.reset_index(drop=True, inplace=True)


# 1.3 Compilation of 1.1 and 1.2
# Output: ["id", "description", "reviews", "review_scores_value"]

data = pd.merge(all_text, compilation_reviews,
    on="id")[['id', 'description', 'reviews', 'review_scores_value']]

data_spot = data.replace(["\n", "\r", "\'"], [" ", " ", ""], regex=True)

data_spot.to_csv("results/data_spot.txt", sep="\t", index=None)

data_spot.sample(n=5000).to_csv("results/data_spot_small.txt", 
    sep="\t", index=None)

'''
------------------------------------------------------------------------------
# 2. UNIT OF ANALYSIS: PERSON
------------------------------------------------------------------------------
'''

# All that a person says
reviews_person = reviews[['reviewer_id', 'comments']]
reviews_person.dropna(inplace=True)

compilation_person = pd.DataFrame(reviews_person.groupby(["reviewer_id"]
    ).apply(lambda x: " ".join(x["comments"]))).rename(
    columns = {0: 'text_person'})
compilation_person['id'] = compilation_person.index
compilation_person.reset_index(drop=True, inplace=True)
compilation_person["char_len"] = compilation_person["text_person"].str.len()
mask = compilation_person["char_len"] > 1800
data_aux_person = compilation_person[mask].reset_index(drop=True).drop(
    columns = ["char_len"], axis = 1)[['id', 'text_person']]

data_person = data_aux_person.replace(
    ["\n", "\r", "\'"], [" ", " ", ""], regex=True)

data_person.to_csv("results/data_person.txt", sep="\t", index=None)

'''
------------------------------------------------------------------------------
# 3. BUILD THE MATRIX OF COUNTS AND PMI [PERSON]
------------------------------------------------------------------------------
'''

# 3.1 BASIC CASE (3000W X 1577W)

# File of context words
# context_file = "data/vocab-25k.txt"
context_file = "data/vocab-3k.txt"
# File of main words
words_file = "data/vocab-wordsim.txt"
# Simple file 3k

# File with the text of a person
person_file = "results/data_person.txt"
name_col_pop = "text_person"
w = 1

counts_df, pmi = create_counts_pmi(w, words_file, context_file, 
    person_file, name_col_pop)

counts_df.to_csv("results/counts_person_basic.txt")
pmi.to_csv("results/pmi_person_basic.txt")


# # 3.2 LARGER CASE (3000W X 3000W)

# counts_df_3000, pmi_3000 = create_counts_pmi(w, context_file, context_file, 
#     person_file, name_col_pop)

# counts_df.to_csv("results/counts_person_3000.txt")
# pmi.to_csv("results/pmi_person_3000.txt")

'''
------------------------------------------------------------------------------
# 4. BUILD THE MATRIX OF COUNTS AND PMI [SPOT]
------------------------------------------------------------------------------
'''

spot_file = "results/data_spot_small.txt"
name_col_spot = "description"


counts_df_spot, pmi_spot = create_counts_pmi(w, words_file, context_file, 
    spot_file, name_col_spot)

counts_df_spot.to_csv("results/counts_spot_basic.txt")
pmi_spot.to_csv("results/pmi_spot_basic.txt")




















