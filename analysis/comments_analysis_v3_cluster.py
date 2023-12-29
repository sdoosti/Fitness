# -*- coding: utf-8 -*-
"""
Created on 12/29/23

@author: Shahryar Doosti

Analyzing youtube comments
(Cluster computing version)
"""

#%%
import pandas as pd
import os
from datetime import date
import re
import numpy as np
import spacy
from spacy.lang.en.stop_words import STOP_WORDS

today = date.today()
today_str = today.strftime("%m%d%y")

PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILE = "merged_comments.csv"
NEWFILE = "comments_activity_motives.csv"

def data_file_path(file_path):
    return os.path.join(os.path.dirname(PATH),"Data",file_path)

file_path = data_file_path(FILE)
comments = pd.read_csv(file_path)
comments.shape
labeled = pd.read_csv(data_file_path(NEWFILE))
comments2 = pd.DataFrame(data={'comment_text': comments.comment_text.tolist() + labeled.comment_text.tolist()})

print('Basic cleaning ...')     
comments = comments2[comments2.comment_text.notnull()].copy()    
data = comments.comment_text.str.lower().values.tolist()
print(len(data))

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

nlp = spacy.load('en_core_web_trf') 

def preprocessing(text, nlp):
    # Process document using Spacy NLP pipeline.
    doc = nlp(text)
    ents = doc.ents  # Named entities.

    # Keep only words (no numbers, no punctuation).
    # Lemmatize tokens, remove punctuation and remove stopwords.
    tokens = [token.lemma_.lower().strip() for token in doc if token.is_alpha and not token.is_stop and token.lemma_ != '-PRON-' and not token.is_punct]

    # Add named entities, but only if they are a compound of more than word.
    #tokens.extend([str(entity) for entity in ents if len(entity) > 1])
    
    return tokens

print('Processing ...')         
processed_docs = []    
for doc in nlp.pip(docs):
    
    #ents = doc.ents  # Named entities.

    # Keep only words (no numbers, no punctuation).
    # Lemmatize tokens, remove punctuation and remove stopwords.
    doc = [token.lemma_.lower().strip() for token in doc if token.is_alpha and not token.is_stop and token.lemma_ != '-PRON-' and not token.is_punct]

    # Add named entities, but only if they are a compound of more than word.
    #doc.extend([str(entity) for entity in ents if len(entity) > 1])
    
    processed_docs.append(doc)
    
docs = processed_docs.copy()
print(len(docs))
#del processed_docs

print('Saving the text ...')  
processed_path = file_path.replace("merged_comments.csv",f"processed_comments_{today_str}.txt")       
with open(processed_path,'w', encoding="utf-8") as f:    
    for doc in docs:
        f.write(','.join(filter(lambda x: x not in ['',' ','[]','[ ]'],doc))+'\n')
        
print('Cleaning Process Completed!')
# %%
