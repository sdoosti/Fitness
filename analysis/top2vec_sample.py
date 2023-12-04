
import os
import pandas as pd
import numpy as np
from top2vec import Top2Vec


DATA_PATH = "/home/doosti@chapman.edu/projects/Fitness/Data/"

# Load the data
"""
processed_file="processed_comments_102423.txt"
comments_file="merged_comments.csv"
comment_length=10
with open(os.path.join(DATA_PATH,processed_file),"r", encoding="utf-8") as f:
    processed_docs = f.readlines()
length = [len(re.sub("\d+", "", x.strip()).split(',')) for x in processed_docs]
comments = cudf.read_csv(os.path.join(DATA_PATH, "merged_comments.csv"))
comments = comments[comments.comment_text.notnull()].copy()
comments['length'] = length
comments['include'] = comments.length > comment_length
comments = comments[comments.include].copy()
docs = comments['comment_text'].to_list()
"""

from sklearn.datasets import fetch_20newsgroups

docs = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))['data']

model = Top2Vec(documents=docs, speed="fast-learn", workers=8, embedding_model='universal-sentence-encoder')