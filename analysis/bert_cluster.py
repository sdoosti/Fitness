# -*- coding: utf-8 -*-
"""
Created on 11/01/23

@author: Shahryar Doosti

BERT Analysis on Youtube Comments
(for cluster computing)
"""

import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from datetime import date
import re
import time
from tqdm import tqdm
import multiprocessing

# get an argument from command line
import sys
threshold = int(sys.argv[1])

PATH = os.path.abspath(os.getcwd())


processed_path = "processed_comments_102423.txt"
comments_path = "merged_comments.csv"

def data_file_path(file_path):
    return os.path.join(os.path.dirname(PATH),"Data",file_path)

with open(data_file_path(processed_path),"r") as f:
    processed_docs = f.readlines()
comments = pd.read_csv(data_file_path(comments_path))
comments = comments[comments.comment_text.notnull()].copy()
comments['processed_text'] = [re.sub("\d+", "", x.strip())for x in processed_docs]
comments['length'] = comments.processed_text.apply(lambda x: len(x.split(',')))
comments['include'] = comments.length > 10
comments = comments[comments.include].copy()

docs = comments.processed_text.to_list()

# BERT
from transformers import AutoModel, AutoTokenizer
# Initialize the BERT tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# Initialize the BERT model
bert_model = AutoModel.from_pretrained('bert-base-uncased')

def vectorize_texts(texts):
    embeddings = []
    for text in tqdm(texts):
        # Encode the text and return tensors
        encoded_inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        # Get the model's output
        output = bert_model(**encoded_inputs)
        # Get the hidden states
        hidden_states = output.last_hidden_state
        # Reshape the tensor and detach it from the current graph
        embeddings.append(hidden_states.view(-1, hidden_states.shape[-1]).detach().numpy())
    return embeddings

# a function to get the embeddings of a text fast
def get_embeddings(text):
    encoded_inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    output = bert_model(**encoded_inputs)
    hidden_states = output.last_hidden_state
    return hidden_states.view(-1, hidden_states.shape[-1]).detach().numpy()

# write the embedding function in parallel
def parallelize_vectorize_texts(texts, num_cores=4):
    pool = multiprocessing.Pool(num_cores)
    embeddings = pool.map(get_embeddings, texts)
    pool.close()
    pool.join()
    return embeddings

a = time.time()
text_embeddings = vectorize_texts(docs[:threshold])
#text_embeddings = [get_embeddings(text) for text in tqdm(docs[:threshold])]
#text_embeddings = parallelize_vectorize_texts(docs[:threshold])
b = time.time()

print(f'Time taken: {b-a} seconds for {threshold} comments')

# with open(data_file_path("bert_embeddings.txt"),"w") as f:
#     f.write(str(text_embeddings))

np.save(data_file_path("bert_embeddings.npy"), np.array(text_embeddings, dtype=object), allow_pickle=True)






"""
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
"""