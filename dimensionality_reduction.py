"""
    File 'dimensionality_reduction.py' consists of function 'data_embedding'
        for dimensionality reduction.
"""
import pandas as pd
import numpy as np
import sklearn.manifold
import sklearn.decomposition


def data_embedding(data, customer_ids, name):
    """
        Using data embedding with 3 different algorithms:
            1. Multidimensional scaling (MDS)
            2. t-distributed Stochastic Neighbor Embedding (t-SNE)
            3. Spectral Embedding
        param:
            1. data - pandas DataFrame (10000, 82), where values are mean
                spendings of customers for every category
            2. customer_ids - numpy array (10000, ) with all customer ids
            3. name - string that represents name of embedding algorithm
        return:
            data_embedded - pandas DataFrame (10000, 3) of embedded data
    """
    # Model selection
    if name == "MDS":
        # MDS
        model_embedding = sklearn.manifold.MDS(n_components=3, n_init=1,
                                               max_iter=2000, n_jobs=-1)
    elif name == "t-SNE":
        # t-SNE
        model_embedding = sklearn.manifold.TSNE(n_components=3,
                                                perplexity=10,
                                                learning_rate=100,
                                                n_iter=2000)
    elif name == "spectral embedding":
        # Spectral Embedding
        model_embedding = \
            sklearn.manifold.SpectralEmbedding(n_components=3, n_jobs=-1)
    else:
        # Raise exception
        raise Exception("Improper method!")

    # Embedding model fit
    data_embedded = model_embedding.fit_transform(data)

    # Column names creation
    feature_names = np.array(["Feature " + str(i) for i in range(3)])

    # DataFrame creation and saving
    data_embedded = pd.DataFrame(data_embedded, columns=feature_names,
                                 index=customer_ids)
    data_embedded.index.name = "customer_id"
    data_embedded.to_csv(path_or_buf="data/output/data_" + name + ".csv")

    return data_embedded
