# -*- coding: utf-8 -*-
"""
Created on 10/24/23

@author: Shahryar Doosti

TF-IDF Analysis on Youtube Comments
"""

import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from datetime import date
import re

today = date.today()
today_str = today.strftime("%m%d%y")

PATH = os.path.abspath(__file__)

processed_path = "processed_comments_102423.txt"
comments_path = "merged_comments.csv"

threshold = 10 # minimum number of terms for each comments (after preprocessing)

def data_file_path(file_path):
    return os.path.join(os.path.dirname(PATH),"Data",file_path)

with open(data_file_path(processed_path),"r") as f:
    processed_docs = f.readlines()
comments = pd.read_csv(data_file_path(comments_path))
comments = comments[comments.comment_text.notnull()].copy()
comments['processed_text'] = [re.sub("\d+", "", x.strip())for x in processed_docs]
comments['length'] = comments.processed_text.apply(lambda x: len(x.split(',')))
comments['include'] = comments.length > threshold

# TF-IDF
docs = comments[comments.include].processed_text.to_list()
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

from sklearn.decomposition import LatentDirichletAllocation

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
doc_topic_dist = lda.transform(tfidf_norm)

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