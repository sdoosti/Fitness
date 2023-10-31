# -*- coding: utf-8 -*-
"""
Created on 10/13/23

@author: Shahryar Doosti

Analyzing youtube comments
"""

#%%
import pandas as pd
import os
from datetime import date


today = date.today()
today_str = today.strftime("%m%d%y")

PATH = os.path.abspath(__file__)
FILE = "merged_comments.csv"

def data_file_path(file_path):
    return os.path.join(os.path.dirname(PATH),"Data",file_path)

file_path = data_file_path(FILE)
comments = pd.read_csv(file_path)
comments.shape

#%%
# goal to exclude the comments left by the creator
videos = pd.read_csv(file_path.replace("merged_comments","videos_101223"))

# matching with the creator id
comments['creator_id'] = comments.merge(videos,how='left',on='video_id').creator_id.values
# identifying the number of videos a user commented for a creator
comments['no_videos_commented'] = comments.groupby(['creator_id','user_id']).comment_text.transform('count').values
# finding the comments left by users with maximum comments per creator
idx = comments.groupby('video_id').no_videos_commented.transform('max') == comments.no_videos_commented
comments[idx].comment_text.sample() # not good. didn't work. picked up many user comments

#%% a different approach

# calculate the max number of videos
comments['no_videos'] = comments.groupby('creator_id').video_id.transform('nunique').values
# summary of number of videos posted by creators
comments.groupby('creator_id').no_videos.first().sort_values()
idx = comments.groupby('video_id').no_videos_commented.transform('max') >= comments.no_videos
comments[idx].comment_text.sample().iloc[0] # this one didn't seem to work either :(
comments.groupby(['creator_id','user_id']).video_id.nunique().hist(bins=30)

#%%
import re
import numpy as np
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from nltk.corpus import stopwords
from collections import Counter

print('Basic cleaning ...')     
comments = comments[comments.comment_text.notnull()].copy()    
data = comments.comment_text.values.tolist()

# not removing emails, hashtags, and urls as they will be removed anyway
# unless they are important

# Remove new line characters
data = [re.sub('\s+', ' ', sent) for sent in data]
               
# Remove 's
data = [re.sub("(\'s)","",sent) for sent in data]
               
# Remove distracting single quotes
data = [re.sub("\'", "", sent) for sent in data]

# Remove extended stop words
# exstopwords = set(extended_stopwords)
# data = [' '.join([w for w in sent.split() if not w.lower() in exstopwords]) for sent in data]

docs = [x.strip() for x in data] 
comments['comment_id'] = np.arange(comments.shape[0])
doc_ids = comments.comment_id.tolist()  

# combined stopwords
swlist = list(STOP_WORDS)
swlist.extend(stopwords.words('english'))
STOP_WORDS = set(swlist)

#%% spacy
nlp = spacy.load('en_core_web_sm')        

print('Processing ...')         
processed_docs = []    
for doc in nlp.pipe(docs):
    # Process document using Spacy NLP pipeline.
    
    ents = doc.ents  # Named entities.

    # Keep only words (no numbers, no punctuation).
    # Lemmatize tokens, remove punctuation and remove stopwords.
    doc = [token.lemma_.lower().strip() for token in doc if token.is_alpha and not token.is_stop  and token.lemma_ != '-PRON-']

    # Add named entities, but only if they are a compound of more than word.
    doc.extend([str(entity) for entity in ents if len(entity) > 1])
    
    processed_docs.append(doc)
    
docs = processed_docs.copy()
print(len(docs))
#del processed_docs

print('Saving the text ...')  
processed_path = file_path.replace("merged_comments.csv","processed_comments_102423.txt")       
with open(processed_path,'w') as f:    
    for doc in docs:
        f.write(','.join(filter(lambda x: x not in ['',' ','[]','[ ]'],doc))+'\n')
        

print('Cleaning Process Completed!')
# %%
