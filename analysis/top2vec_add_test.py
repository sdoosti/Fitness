
"""
Modified to add the labeled data to the model trained previously.

(Only runs in lower case mode)
"""
import os
import pandas as pd
import numpy as np
from top2vec import Top2Vec
import sys
import re


# Load the data
DATA_PATH = "/home/doosti@chapman.edu/projects/Fitness/Data/"
processed_file="processed_comments_102423.txt"
comments_file="merged_comments.csv"
labeled = "comments_activity_motives.csv"

labeled = pd.read_csv(os.path.join(DATA_PATH, labeled))
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

print("Data is loaded.")

model_name = "top2vec_lowercase_learn_doc2vec.model"
model_path = os.path.join(DATA_PATH, model_name)
model = Top2Vec.load(model_path)

print("Top2Vec model is loaded.")

# Add the labeled data to the model
model.add_documents(labeled.comment_text.tolist())#, labeled.comment_id.tolist())
print("Labeled data is added to the model.")

# Save the model
model.save(os.path.join(DATA_PATH, "top2vec_lowercase_learn_doc2vec_add_labeled.model"))
print("Model is saved.")

# Save the document vectors
topic_nums, topic_scores, topic_words, word_scores = model.get_documents_topics(list(range(comments.shape[0])), num_topics=5)