# -*- coding: utf-8 -*-

"""
Created at 12/1/23
by Shahryar Doosti

This module implements BERTopic model for topic modeling on YouTube comments.
"""

import os
import pandas as pd
import numpy as np

from bertopic import BERTopic

DATA_PATH = "/home/doosti@chapman.edu/projects/Fitness/Data/"

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
    comments = cudf.read_csv(os.path.join(DATA_PATH, "merged_comments.csv"))
    comments = comments[comments.comment_text.notnull()].copy()
    comments['length'] = length
    comments['include'] = comments.length > comment_length
    comments = comments[comments.include].copy()
    return comments

def bert_topic(docs):
    """ This function implements BERTopic model for topic modeling.
       docs (list): list of documents
       return (tuple): topic model and topics """
    model = BERTopic(language="english", verbose=True)
    topics, _ = model.fit_transform(docs)
    return model, topics