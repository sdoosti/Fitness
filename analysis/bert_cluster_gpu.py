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
from datetime import date
import re
import time
from tqdm import tqdm
import torch

# get an argument from command line
import sys
#threshold = int(sys.argv[1])

PATH = os.path.abspath(os.getcwd())

# gpu device
device = "cuda:0" if torch.cuda.is_available() else "cpu"

print(f"using {device}")

# today's date in format of DDMMYY string
today_str = date.today().strftime("%d%m%y")

# load the BERT model
from transformers import AutoModel, AutoTokenizer
# Initialize the BERT tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
# Initialize the BERT model
bert_model = AutoModel.from_pretrained('bert-base-uncased')
# Move the model to the GPU
bert_model = bert_model.to(device)

def data_file_path(file_path):
    return os.path.join(os.path.dirname(PATH),"Data",file_path)

def load_data_files(processed_path="processed_comments_102423.txt", comments_path="merged_comments.csv", comment_length=10):
    """ This function loads the processed comments and the original comments and returns the processed comments and the original comments in a dataframe.
       processed_path (str): processed comments file name
       comments_path (str): original comments file name
       comments_length (int): minimum number of words in a comment
       return (pandas.DataFrame, list): processed comments and original comments in a dataframe  """
    with open(data_file_path(processed_path),"r", encoding="utf-8") as f:
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
    encoded_inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True).to(device)
    # Get the model's output
    with torch.no_grad():
        output = bert_model(**encoded_inputs)
    # Get the hidden states
    hidden_states = output.last_hidden_state
    # Reshape the tensor and detach it from the current graph
    embeddings = hidden_states.view(-1, hidden_states.shape[-1]).detach().cpu().numpy()
    return embeddings

def get_embeddings_batch(texts, embeddings='word', last_hidden = True, batch_size=1000, max_length=50):
    """ This function extracts the embeddings of the texts using BERT model in batch mode (multiple documents)
        texts (list): list of texts to be embedded
        embeddings (str): ['word', 'sentence', 'pooled']
            'word': returns the word embeddings of the text
            'sentence': returns the sentence embedding of the text represented by [CLS] token
            'pooled': returns the pooled output of the text by average word embdeddings.
        last_hidden (bool): if True, the function returns the last hidden state of the text, if False, the function returns the either word embeddings or pooled output of the text.
        batch_size (int): number of documents to be embedded in each batch
        max_length (int): maximum length of the documents in the batch mode
        return (numpy.ndarray): embeddings of the texts
    """
    docs_embeddings = []
    for idx in tqdm(range(0, len(texts), batch_size)):
        batch = texts[idx : min(len(texts), idx+batch_size)]
        encoded = tokenizer.batch_encode_plus(batch,max_length=max_length, padding='max_length', truncation=True)
        encoded = {key:torch.LongTensor(value).to(device) for key, value in encoded.items()}
        if not last_hidden:
            encoded['output_hidden_states'] = True
        with torch.no_grad():
            outputs = bert_model(**encoded)
        if last_hidden:
            outputs = outputs.last_hidden_state
        else:
            # selecting the last 4 layers
            layers = [-4,-3,-2,-1]
            # Get all hidden states
            states = outputs.hidden_states
            # Stack and sum all requested layers
            outputs = torch.stack([states[i] for i in layers]).sum(0).squeeze()

        if embeddings == 'word':
            docs_embeddings.append(outputs.detach().cpu().numpy())
        elif embeddings == 'sentence':
            cls_embeddings = outputs[:,0,:]
            docs_embeddings.append(cls_embeddings.detach().cpu().numpy())
        else:
            attention = encoded['attention_mask'].reshape((outputs.size()[0], outputs.size()[1], -1)).expand(-1, -1, 768)
            pooled_embeddings = torch.mul(outputs, attention)
            denominator = torch.count_nonzero(pooled_embeddings, dim=1)
            summation = torch.sum(embeddings, dim=1)
            mean_embeddings = torch.div(summation, denominator)
            docs_embeddings.append(mean_embeddings.detach().cpu().numpy())
    return np.vstack(docs_embeddings)

def save_embeddings(text_embeddings, embedding_format = 'word', optional_tag=''):
    np.save(data_file_path(f"bert_embeddings_{len(text_embeddings)}docs_{embedding_format}_{optional_tag}{today_str}.npy"), np.array(text_embeddings, dtype=object), allow_pickle=True)

def main(embedding_format = 'word'):
    # load data
    comments, docs = load_data_files()
    # get embeddings
    text_embeddings = get_embeddings_batch(docs, embeddings=embedding_format, max_length=100, batch_size=1000)
    save_embeddings(text_embeddings, embedding_format)
    return text_embeddings

if __name__ == "__main__":
    text_embeddings = main()
    print(text_embeddings.shape)