import cudf
from cuml.cluster import KMeans
from cuml.manifold import TSNE
from cuml.metrics import silhouette_score

import numpy as np
import pandas as pd
import os
import re
import numba.cuda

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

# Load the embeddings
def load_embeddings(version='tokens', gpu=True):
    """ This function loads the embeddings and returns them as a numpy array.
       version (str): lowercase, original, or tokens
       gpu (bool): if True, the embeddings are loaded on the GPU
       return (numpy.ndarray or cuda.DataFrame): embeddings as a numpy array  """
    embed_file = f"bert_embeddings_221979docs_sentence_{version}_071123.npy"
    embeddings = np.load(os.path.join(DATA_PATH, embed_file), allow_pickle=True)
    if gpu:
        #embeddings = numba.cuda.to_device(embeddings)
        embd_dict = {'fea%d'%i:embeddings[:,i] for i in range(embeddings.shape[1])}
        return cudf.DataFrame(embd_dict)
    else:
        return embeddings

# Kmeans clustering
def kmeans_clustering(data, n_clusters=20):
    """ This function performs kmeans clustering on the embeddings and returns the cluster labels.
       data (cudf.DataFrame): embeddings
       n_clusters (int): number of clusters
         return (numpy.ndarray): cluster labels, silhouette score  """
    # Ensure the data is in the format expected by cuml functions
    if not isinstance(data, cudf.DataFrame):
        data = cudf.DataFrame(data)
    # Create a KMeans model
    model = KMeans(n_clusters=n_clusters)
    # Fit the model to the data
    model.fit(data)
    # Predict the clusters
    labels = model.predict(data)
    # Calculate silhouette score
    score = silhouette_score(data, labels)
    return labels, score

def save_cluster_labels(comments, labels, version='tokens'):
    """ This function saves the cluster labels in a csv file.
       comments (cudf.DataFrame): original comments
       labels (numpy.ndarray): cluster labels
       version (str): lowercase, original, or tokens  """
    comments['cluster'] = labels
    comments.to_csv(os.path.join(DATA_PATH, f"comments_cluster_{version}.csv"),index=False)

def calculate_cluster_centers(embeddings, labels):
    """ This function calculates the cluster centers.
       embeddings (cudf.DataFrame): embeddings
       labels (numpy.ndarray): cluster labels
       return (cudf.DataFrame): cluster centers  """
    # Ensure the data is in the format expected by cuml functions
    if not isinstance(embeddings, cudf.DataFrame):
        embeddings = cudf.DataFrame(embeddings)
    # Add the cluster labels to the embeddings
    embeddings['cluster'] = labels
    # Calculate the cluster centers
    centers = embeddings.groupby('cluster').mean()
    return centers

def calculate_cluster_sizes(labels):
    """ This function calculates the cluster sizes.
       labels (numpy.ndarray): cluster labels
       return (cudf.DataFrame): cluster sizes  """
    # Ensure the data is in the format expected by cuml functions
    if not isinstance(labels, cudf.DataFrame):
        labels = cudf.DataFrame(labels)
    # Calculate the cluster sizes
    sizes = labels.groupby('cluster').size()
    return sizes

def tsne_calc(data,perplexity=50,n_neighbors=500,version='tokens',save=True):  
    """ This function calculates the t-SNE embeddings.
       data (cudf.DataFrame): embeddings
       perplexity (int): perplexity
       n_neighbors (int): number of neighbors
       return (cudf.DataFrame): t-SNE embeddings  """
    # Ensure the data is in the format expected by cuml functions
    if not isinstance(data, cudf.DataFrame):
        data = cudf.DataFrame(data)
    # Create a TSNE model
    tsne = TSNE(perplexity=perplexity, n_neighbors=n_neighbors)
    # Fit the model to the data and transform it
    df_transformed = tsne.fit_transform(data)
    if save:
        df_transformed.to_csv(os.path.join(DATA_PATH, f"tsne_{version}_perp{perplexity}_n{n_neighbors}.csv"),index=False)
    return df_transformed


