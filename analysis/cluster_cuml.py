import cudf
from cuml.cluster import KMeans
from cuml.manifold import TSNE
import numpy as np
import pandas as pd
import os
import re
import numba.cuda

DATA_PATH = "/home/doosti@chapman.edu/projects/Fitness/Data/"

# Load the data
with open(os.path.join(DATA_PATH,"processed_comments_102423.txt"),"r", encoding="utf-8") as f:
    processed_docs = f.readlines()
length = [len(re.sub("\d+", "", x.strip()).split(',')) for x in processed_docs]
comments = cudf.read_csv(os.path.join(DATA_PATH, "merged_comments.csv"))
comments = comments[comments.comment_text.notnull()].copy()
comments['length'] = length
comments['include'] = comments.length > 10
comments = comments[comments.include].copy()
print(comments.shape)

# Load the embeddings
version = 'tokens' # lowercase, original, or tokens
embed_file = f"bert_embeddings_221979docs_sentence_{version}_071123.npy"

embeddings = np.load(os.path.join(DATA_PATH, embed_file), allow_pickle=True)
#gpu = numba.cuda.to_device(embeddings)
embd_dict = {'fea%d'%i:embeddings[:,i] for i in range(embeddings.shape[1])}
print(len(embd_dict))
df = cudf.DataFrame(embd_dict)

print(df.shape)

print("Kmeans Clustering")
# Specify the number of clusters
n_clusters = 20

# Create a KMeans model
model = KMeans(n_clusters=n_clusters)

# Fit the model to the data
model.fit(df)

# Predict the clusters
comments['cluster'] = model.predict(df)

print(comments.head())

comments.to_csv(os.path.join(DATA_PATH, f"comments_cluster_{version}.csv"),index=False)

print("t-SNE Analysis")
# Specify the number of components
n_components = 2

# Create a TSNE model
perplexity = 50
n_neighbors = 500
tsne = TSNE(n_components=n_components, perplexity=perplexity, n_neighbors=n_neighbors)

# Fit the model to the data and transform it
df_transformed = tsne.fit_transform(df)

print(df_transformed.head())
df_transformed.to_csv(os.path.join(DATA_PATH, f"tsne_{version}_perp{perplexity}_n{n_neighbors}.csv"),index=False)



