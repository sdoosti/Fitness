
import os
import pandas as pd
import numpy as np
from top2vec import Top2Vec
import sys
import re

DATA_PATH = "/home/doosti@chapman.edu/projects/Fitness/Data/"

# instert model type from the command line argument
data = sys.argv[1]
# add assert message
assert data in ['lowercase', 'tokens'] # model type should be either lowercase or tokens
if len(sys.argv) > 2:
    assert sys.argv[2] in ['fast-learn', 'learn', 'deep-learn'] # speed should be either fast-learn, learn or deep-learn
    speed = sys.argv[2]
else:
    speed = 'learn'

if len(sys.argv) > 3:
    assert sys.argv[3] in ['doc2vec','universal-sentence-encoder', 'distiluse-base-multilingual-cased-v2', 'distilbert-base-nli-mean-tokens'] # embedding model should be either universal-sentence-encoder, distiluse-base-multilingual-cased-v2 or distilbert-base-nli-mean-tokens
    embedding_model = sys.argv[3]
else:
    embedding_model = 'doc2vec'

# print the model setup
print('-'*50)
print('Model setup:')
print(f"Model type: {data}")
print(f"Speed: {speed}")
print(f"Embedding model: {embedding_model}")
print('-'*50)

# Load the data
processed_file="processed_comments_102423.txt"
comments_file="merged_comments.csv"
comment_length=10
with open(os.path.join(DATA_PATH,processed_file),"r", encoding="utf-8") as f:
    processed_docs = f.readlines()
length = [len(re.sub("\d+", "", x.strip()).split(',')) for x in processed_docs]
comments = pd.read_csv(os.path.join(DATA_PATH, "merged_comments.csv"))
comments = comments[comments.comment_text.notnull()].copy()
comments['processed'] = processed_docs
comments['length'] = length
comments['include'] = comments.length > comment_length
comments = comments[comments.include].copy()
comments['comment_text'] = comments['comment_text'].str.lower()

if data == 'lowercase':
    docs = comments['comment_text'].to_list()
else:
    docs = comments['processed'].to_list()

model = Top2Vec(documents=docs, speed=speed, workers=8, embedding_model=embedding_model, keep_documents=True)
model.save(os.path.join(DATA_PATH, f"top2vec_{data}_{speed}_{embedding_model}.model"))

print("Top2Vec model is saved.")