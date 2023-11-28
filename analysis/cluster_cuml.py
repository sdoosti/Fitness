import cudf
from cuml.cluster import KMeans
from cuml.manifold import TSNE
from cuml.metrics.cluster.silhouette_score import cython_silhouette_score as silhouette_score
from cuml import DBSCAN
#from cuml.metrics.cluster import adjusted_rand_score

# import RecurssionError
import sys
sys.setrecursionlimit(100000)

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
        #TODO: replace this with cupy array
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
    return labels

# DBSCAN clustering
def dbscan_clustering(data, eps = 4.4, min_samples = 1000):
    """ This function performs dbscan clustering on the embeddings and returns the cluster labels.
       data (cudf.DataFrame): embeddings
       eps (float): epsilon
       min_samples (int): minimum number of samples
         return (numpy.ndarray): cluster labels  """
    # Ensure the data is in the format expected by cuml functions
    if not isinstance(data, cudf.DataFrame):
        data = cudf.DataFrame(data)
    # Create a DBSCAN model
    model = DBSCAN(eps=eps, min_samples=min_samples)
    # Fit the model to the data
    model.fit(data)
    # Predict the clusters
    labels = model.labels_
    return labels

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

def main(method='kmeans', version= 'tokens', save=False, details=False, sil_score=False):
    """ This function performs clustering on the embeddings and prints the results.
       method (str): clustering method (kmeans or dbscan)
       save (bool): if True, the cluster labels are saved in a csv file
       details (bool): if True, the cluster centers, cluster sizes, and t-SNE embeddings are printed"""
    # Load the data
    comments = load_data_file()
    # Load the embeddings
    embeddings = load_embeddings(version=version)
    # Perform clustering
    if method == 'kmeans':
        labels = kmeans_clustering(embeddings)
    elif method == 'dbscan':
        labels = dbscan_clustering(embeddings)
    # Save the cluster labels
    if save:
        save_cluster_labels(comments, labels, version)
    
    if sil_score:
        # Calculate the silhouette score
        score = silhouette_score(embeddings, labels)

    if details:
        # Calculate the cluster centers
        centers = calculate_cluster_centers(embeddings, labels)
        # Calculate the cluster sizes
        sizes = calculate_cluster_sizes(labels)
        # Calculate the t-SNE embeddings
        tsne = tsne_calc(embeddings)
        print(f"silhouette score: {score}")
        print(f"cluster sizes: {sizes}")
        print(f"cluster centers: {centers}")
        print(f"t-SNE embeddings: {tsne}")

def parameters_search(method='kmeans', n_samples = 10000, version = 'tokens'):
    # this function explores the parameters of kmeans clustering
    # Load the data
    print("Loading the data...")
    comments = load_data_file()
    # Load the embeddings
    print("Loading the embeddings...")
    embeddings = load_embeddings(version=version)
    if method == 'kmeans':
        # Perform kmeans clustering
        print("Performing kmeans clustering...")
        for n_clusters in range(2, 20):
            labels = kmeans_clustering(embeddings, n_clusters=n_clusters)
            # Calculate the silhouette score
            # select a random sample of 10000 rows (return index)
            sample = embeddings.sample(n=1000).index
            score = silhouette_score(embeddings.loc[sample], labels[sample])
            print(f"clusters: {n_clusters}, silhouette score: {score}")

    if method == 'dbscan':
        # Perform dbscan clustering
        print("Performing dbscan clustering...")
        for eps in range(20,30):
            for min_samples in [1000,2000,5000]:
                if (eps != 22) or (min_samples != 1000):
                    continue
                labels = dbscan_clustering(embeddings, eps=eps/5, min_samples=min_samples)
                print('-'*50)
                print('eps: ', eps/5, 'min_samples: ', min_samples)
                print(labels.value_counts())
                if len(labels.value_counts()) < 3:
                    continue
                # Calculate the silhouette score
                # select a random sample of 10000 rows (return index)
                sample = embeddings.sample(n=n_samples).index
                # print(labels[sample].head(20))
                score = silhouette_score(embeddings.loc[sample], labels[sample])
                print(f"eps: {eps}, min_samples: {min_samples}, silhouette score: {score}")

if __name__ == "__main__":
    # get the arguments from the command line
    function = sys.argv[1]
    method = sys.argv[2]
    version = sys.argv[3]
    if function == 'main':
        print('Running main function')
        # if there is argument take it as input, otherwise use the default value
        if len(sys.argv) > 4:
            save = sys.argv[4]
        else:
            save = False
        if len(sys.argv) > 5:
            details = sys.argv[5]
        else:
            details = False
        if len(sys.argv) > 6:
            sil_score = sys.argv[6]
        else:   
            sil_score = False
        main(method=method, version=version, save=save, details=details, sil_score=sil_score)
    elif function == 'search':
        print('Running parameters_search function')
        parameters_search(method=method, version=version)
    else:
        print("Invalid function name")
    #parameters_search(method='kmeans',version='lowercase')
