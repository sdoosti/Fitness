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
import torch

# get an argument from command line
import sys
#threshold = int(sys.argv[1])

PATH = os.path.abspath(os.getcwd())

# load the BERT model
from transformers import AutoModel, AutoTokenizer
# Initialize the BERT tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
# Initialize the BERT model
bert_model = AutoModel.from_pretrained('bert-base-uncased')


def data_file_path(file_path):
    return os.path.join(os.path.dirname(PATH),"Data",file_path)

def load_data_files(processed_path="processed_comments_102423.txt", comments_path="merged_comments.csv", comment_length=10):
    """ This function loads the processed comments and the original comments and returns the processed comments and the original comments in a dataframe.
       processed_path (str): processed comments file name
       comments_path (str): original comments file name
       comments_length (int): minimum number of words in a comment
       return (pandas.DataFrame, list): processed comments and original comments in a dataframe  """
    with open(data_file_path(processed_path),"r") as f:
        processed_docs = f.readlines()
    comments = pd.read_csv(data_file_path(comments_path))
    comments = comments[comments.comment_text.notnull()].copy()
    comments['processed_text'] = [re.sub("\d+", "", x.strip())for x in processed_docs]
    comments['length'] = comments.processed_text.apply(lambda x: len(x.split(',')))
    if comment_length is not None:
        comments['include'] = comments.length > comment_length
        comments = comments[comments.include].copy()
    docs = comments.processed_text.to_list()
    return comments, docs

def get_embeddings(text, sentence_embeddings=True, last_hidden = True):
    """This function extracts the embeddings of the text using BERT model.  It returns the embeddings of the text.
       text (str): text to be embedded
       sentence_embeddings (bool): if True, the function returns the sentence embedding of the text, if False, the function returns the word embeddings of the text.
       last_hidden (bool): if True, the function returns the last hidden state of the text, if False, the function returns the pooled output of the text.
       return (numpy.ndarray): embeddings of the text"""
    # Encode the text and return tensors
    encoded_inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    # Get the model's output
    with torch.no_grad():
        output = bert_model(**encoded_inputs)
    # Get the hidden states
    hidden_states = output.last_hidden_state
    # Reshape the tensor and detach it from the current graph
    embeddings = hidden_states.view(-1, hidden_states.shape[-1]).detach().numpy()
    return embeddings

for idx in range(0, len(docs[:500]), 100):
    batch = docs[idx : min(len(docs[:500]), idx+100)]
    # encoded = tokenizer(batch)
    encoded = tokenizer.batch_encode_plus(batch,max_length=100, padding='max_length', truncation=True)
    encoded = {key:torch.LongTensor(value) for key, value in encoded.items()}
    with torch.no_grad():
        outputs = bert_model(**encoded)
    print(outputs.last_hidden_state.size())
    lhs = outputs.last_hidden_state
    attention = encoded['attention_mask'].reshape((lhs.size()[0], lhs.size()[1], -1)).expand(-1, -1, 768)
    embeddings = torch.mul(lhs, attention)
    denominator = torch.count_nonzero(embeddings, dim=1)
    summation = torch.sum(embeddings, dim=1)
    mean_embeddings = torch.div(summation, denominator)


def find_similar_comments(text_embeddings, threshold=0.9):
    similar_comments = []
    for i, embedding in enumerate(text_embeddings):
        # find similarity of the embedding with all other embeddings
        similarities = []
        for j, other_embedding in enumerate(text_embeddings):
            if i != j:
                similarities.append(cosine_similarity(embedding.mean(0).reshape(1,-1), other_embedding.mean(0).reshape(1,-1))[0][0])
            else:
                similarities.append(0)
        # find the indexes of similar embeddings
        print(similarities)
        most_similar_index = np.argmax(np.array(similarities))
        # add the indexes to the list
        similar_comments.append(most_similar_index)
    return similar_comments

# write the embedding function in parallel
def parallelize_vectorize_texts(texts, num_cores=4):
    pool = multiprocessing.Pool(num_cores)
    embeddings = pool.map(get_embeddings, texts)
    pool.close()
    pool.join()
    return embeddings


text_embeddings = np.load(data_file_path("bert_embeddings.npy"), allow_pickle=True)

# find similar comments
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cosine
# find cluster of similar comments using text embeddings
def find_similar_comments(text_embeddings, threshold=0.9):
    similar_comments = []
    for i, embedding in enumerate(text_embeddings):
        # find similarity of the embedding with all other embeddings
        similarities = []
        for j, other_embedding in enumerate(text_embeddings):
            if i != j:
                similarities.append(cosine_similarity(embedding.mean(0).reshape(1,-1), other_embedding.mean(0).reshape(1,-1))[0][0])
            else:
                similarities.append(0)
        # find the indexes of similar embeddings
        most_similar_index = np.argmax(np.array(similarities))
        # add the indexes to the list
        similar_comments.append(most_similar_index)
    return similar_comments


a = time.time()
text_embeddings = vectorize_texts(docs[:threshold])
#text_embeddings = [get_embeddings(text) for text in tqdm(docs[:threshold])]
#text_embeddings = parallelize_vectorize_texts(docs[:threshold])
b = time.time()

print(f'Time taken: {b-a} seconds for {threshold} comments')

# with open(data_file_path("bert_embeddings.txt"),"w") as f:
#     f.write(str(text_embeddings))

np.save(data_file_path("bert_embeddings.npy"), np.array(text_embeddings, dtype=object), allow_pickle=True)

def main():
    # load data
    comments, docs = load_data_files()
    # get embeddings
    text_embeddings = vectorize_texts(docs)
    # find similar comments
    similar_comments = find_similar_comments(text_embeddings)
    # add similar comments to the dataframe
    comments['similar_comment'] = similar_comments
    # save the dataframe
    comments.to_csv(data_file_path("similar_comments.csv"), index=False)

if __name__ == "__main__":
    main()


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