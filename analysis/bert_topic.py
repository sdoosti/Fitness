# -*- coding: utf-8 -*-
"""
Created on 10/26/23

@author: Shahryar Doosti

BERT Analysis on Youtube Comments
"""

import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from datetime import date
import re

today = date.today()
today_str = today.strftime("%m%d%y")

# Set the path
PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(PATH, "Data")

# Load the data
def load_data_file(processed_file="processed_comments_102423.txt", comments_file="merged_comments.csv", comment_length=10):
    """ This function loads the processed comments and the original comments and returns the processed comments and the original comments in a dataframe.
       processed_file (str): processed comments file name
       comments_file (str): original comments file name
       comments_length (int): minimum number of words in a comment
       return (cudf.DataFrame, list): processed comments and original comments in a dataframe  """
    with open(os.path.join(DATA_PATH,processed_file),"r", encoding="utf-8") as f:
        processed_docs = f.readlines()
    length = [len(re.sub("\d+", "", x.strip()).split(',')) for x in processed_docs]
    comments = pd.read_csv(os.path.join(DATA_PATH, "merged_comments.csv"))
    comments = comments[comments.comment_text.notnull()].copy()
    comments['length'] = length
    comments['include'] = comments.length > comment_length
    comments = comments[comments.include].copy()
    return comments


# Load the embeddings
def load_embeddings(version='tokens'):
    """ This function loads the embeddings and returns them as a numpy array.
       version (str): lowercase, original, or tokens
       return (numpy.ndarray): embeddings as a numpy array  """
    embed_file = f"bert_embeddings_221979docs_sentence_{version}_071123.npy"
    embeddings = np.load(os.path.join("E:/", embed_file), allow_pickle=True)
    return embeddings

if __name__ == "__main__":
    comments = load_data_file()
    embeddings = load_embeddings(version='tokens')
    print(comments.head())
    print(embeddings.shape)
    print(today_str)




docs = comments.processed_text.to_list()

# TF-IDF
# create object
tfidf = TfidfVectorizer(max_features=20000, min_df = 0.001, max_df=0.1)
 
# get tf-df values
result = tfidf.fit_transform([' '.join(sent.split(',')) for sent in docs])

# get indexing
print('\nWord indexes:')
print(tfidf.vocabulary_)

# display tf-idf values
print('\ntf-idf value:')
print(result)

# in matrix form
print('\ntf-idf values in matrix form:')
print(result.toarray())

tfidf.get_feature_names()

cleaned_docs = []
for doc in [sent.split(',') for sent in docs]:
    cleaned_words = [word for word in doc if word in tfidf.vocabulary_]
    cleaned_docs.append(cleaned_words)

# topic model
from sklearn.preprocessing import Normalizer
normalizer = Normalizer(norm='l2')
tfidf_norm = normalizer.transform(result)
tfidf_norm = (tfidf_norm + 1) / 2 # to avoid negative values

from sklearn.decomposition import LatentDirichletAllocation

lda = LatentDirichletAllocation(n_components=10) 
lda.fit(tfidf_norm)

doc_topic_dist = lda.transform(tfidf_norm)

# Get the fitted LDA model
lda = LatentDirichletAllocation(n_components=10)
lda.fit(tfidf_norm) 

# Get the word-topic distribution matrix
topic_word_dist = lda.components_  

# Top 10 words per topic
n_top_words = 10
feature_names = tfidf.get_feature_names()

for topic_idx, topic in enumerate(topic_word_dist):
    top_word_idxs = topic.argsort()[:-n_top_words - 1:-1]
    top_words = [feature_names[i] for i in top_word_idxs]
    print(f"Topic {topic_idx}: {', '.join(top_words)}")


# top topics for comments
n_top_topics = 2
for i, dists in enumerate(doc_topic_dist):
    topic_idx = dists.argsort()[:-n_top_topics-1:-1]
    top_topics = [topic_idx[i] for i in range(n_top_topics)]
    print(f"Document {i}: {top_topics}")
    break    

# topic distribution
topic_df = pd.DataFrame(doc_topic_dist, columns=[f"topic_{x}" for x in range(10)])
topic_df['dominant_topic'] = topic_df.idxmax(1).values
topic_df['dom_topic_weight'] = topic_df.apply(lambda x: x.loc[x.dominant_topic], axis=1)
topic_df.dominant_topic.value_counts()
topic_df.dom_topic_weight.describe()
topic_df = topic_df[topic_df.dom_topic_weight>0.1].copy()
# removing non-english comments

print(today_str)