"""
Name:       Raiffeisenbank clients data clustering
Purpose:    Project to test different clustering methods:
                1. DBSCAN algorithm
                2. OPTICS algorithm
                3. k-means algorithm
                4. Clustering by LDA topics
                5. Gaussian mixture model with Bayes classifier clustering
            With different data:
                1. Raw data
                2. MDS transformed data
                3. t-SNE transformed data
                4. Spectral Embedding transformed data
Author:     Artem "Manneq" Arkhipov
Created:    13/01/2020
"""
import time
import data_import
import clustering
import dimensionality_reduction


"""
    File 'main.py' is main file that controls the sequence of function calls.
"""


def importing_data():
    """
        Function to:
            1. load data
            2. Preprocess data
            3. Plot data distributions
            4. Vectorize data for clustering and topic modelling
        return:
            1. data_for_clustering - pandas DataFrame (10000, 82), where
                values are mean spendings of customers for every category
            2. data_for_lda - pandas DataFrame (10000, 82), where values are
                the transactions number of customers for every category
            3. customer_ids - numpy array (10000, ) with all customer ids
            4. category_names - numpy array (82, ) with names of categories
    """
    # Data importing and preprocessing
    print("\t0. Loading and preprocessing data")
    time_start = time.time()

    data, customer_ids, category_names = data_import.preprocessing_data()

    time_end = (time.time() - time_start) / 60
    print("\t   Done. With time " + str(time_end) + " min")

    # Data distribution finding and plotting
    print("\t1. Finding dataset distributions")
    time_start = time.time()

    data_import.data_distributions(data, customer_ids, category_names)

    time_end = (time.time() - time_start) / 60
    print("\t   Images folder: plots/data distributions/")
    print("\t   Done. With time " + str(time_end) + " min")

    # Data vectorization
    print("\t2. Vectorizing dataset")
    time_start = time.time()

    data_for_clustering, data_for_lda = \
        data_import.data_vectorization(data, customer_ids, category_names)

    time_end = (time.time() - time_start) / 60
    print("\t   Data folder: data/output/")
    print("\t   Done. With time " + str(time_end) + " min")

    return data_for_clustering, data_for_lda, customer_ids, category_names


def initial_data_clustering(data_for_clustering, data_for_lda, customer_ids,
                            category_names):
    """
        Function to cluster raw dataset using:
            1. DBSCAN algorithm
            2. OPTICS algorithm
            3. k-means algorithm
            4. Clustering by LDA topics
            5. Gaussian mixture model with Bayes classifier
        param:
            1. data_for_clustering - pandas DataFrame (10000, 82), where
                values are the mean spendings of customers for every
                category
            2. data_for_lda - pandas DataFrame (10000, 82), where values are
                the transactions number of customers for every category
            3. customer_ids - numpy array (10000, ) with all customer ids
            4. category_names - numpy array (82, ) with names of categories
    """
    # Clustering using DBSCAN algorithm
    print("\t0. DBSCAN clustering")
    time_start = time.time()

    clustering.dbscan_clustering(data_for_clustering, "initial data")
    data_for_clustering = data_for_clustering.drop(columns='cluster')

    time_end = (time.time() - time_start) / 60
    print("\t   Images folder: plots/clustering/initial data/DBSCAN/")
    print("\t   Done. With time " + str(time_end) + " min")

    # Clustering using OPTICS algorithm
    print("\t1. OPTICS clustering")
    time_start = time.time()

    clustering.optics_clustering(data_for_clustering, "initial data")
    data_for_clustering = data_for_clustering.drop(columns='cluster')

    time_end = (time.time() - time_start) / 60
    print("\t   Images folder: plots/clustering/initial data/OPTICS/")
    print("\t   Done. With time " + str(time_end) + " min")

    # Clustering using k-means algorithm with
    # Silhouette method for clusters number choosing
    print("\t2. k-means clustering (Silhouette method)")
    time_start = time.time()

    clustering.kmeans_clustering(data_for_clustering, "initial data",
                                 "k-means with Silhouette",
                                 clusters_number=clustering.silhouette_method(
                                     data_for_clustering,
                                     "clustering/initial data/k-means "
                                     "with Silhouette"))
    data_for_clustering = data_for_clustering.drop(columns='cluster')

    time_end = (time.time() - time_start) / 60
    print("\t   Images folder: plots/clustering/initial data/"
          "k-means with Silhouette/")
    print("\t   Done. With time " + str(time_end) + " min")

    # Clustering using k-means algorithm with
    # Elbow (Knee) method for clusters number choosing
    print("\t3. k-means clustering (Elbow (Knee) method)")
    time_start = time.time()

    clustering.kmeans_clustering(data_for_clustering, "initial data",
                                 "k-means with Elbow",
                                 clusters_number=clustering.elbow_method(
                                     data_for_clustering,
                                     "clustering/initial data/k-means "
                                     "with Elbow"))
    data_for_clustering = data_for_clustering.drop(columns='cluster')

    time_end = (time.time() - time_start) / 60
    print("\t   Images folder: plots/clustering/initial data/"
          "k-means with Elbow/")
    print("\t   Done. With time " + str(time_end) + " min")

    # Topic modeling using LDA
    print("\t4. Performing LDA")
    time_start = time.time()

    lda_results, topics_number = \
        clustering.lda_performing(data_for_lda, customer_ids, category_names)

    time_end = (time.time() - time_start) / 60
    print("\t   Data folder: data/output/")
    print("\t   Images folder: plots/LDA/")
    print("\t   Done. With time " + str(time_end) + " min")

    # Clustering using k-means algorithm with LDA topics
    print("\t5. k-means clustering using LDA topics")
    time_start = time.time()

    clustering.kmeans_clustering(data_for_clustering, "initial data",
                                 "k-means with LDA",
                                 clusters_number=topics_number,
                                 auxiliary_data=lda_results.drop(
                                     columns=["dominant_topic"]))
    data_for_clustering = data_for_clustering.drop(columns='cluster')

    time_end = (time.time() - time_start) / 60
    print("\t   Images folder: plots/clustering/initial data/"
          "k-means with LDA/")
    print("\t   Done. With time " + str(time_end) + " min")

    # Clustering using dominant topics as cluster labels
    print("\t6. Clustering using dominant LDA topics")
    time_start = time.time()

    clustering.topic_clustering(data_for_clustering, lda_results,
                                topics_number)
    data_for_clustering = data_for_clustering.drop(columns='dominant_topic')

    time_end = (time.time() - time_start) / 60
    print("\t   Images folder: plots/clustering/initial data/"
          "clustering by LDA topics/")
    print("\t   Done. With time " + str(time_end) + " min")

    # Clustering using Gaussian mixture model with Bayes classifier
    print("\t7. Gaussian mixture clustering")
    time_start = time.time()

    clustering.gaussian_mixture(data_for_clustering, "initial data")
    data_for_clustering = data_for_clustering.drop(columns='cluster')

    time_end = (time.time() - time_start) / 60
    print("\t   Images folder: plots/clustering/initial data/"
          "gaussian mixture/")
    print("\t   Done. With time " + str(time_end) + " min")

    return


def embedded_data_clustering(data_for_clustering, customer_ids, name):
    """
        Function to cluster raw dataset using:
            1. DBSCAN algorithm
            2. OPTICS algorithm
            3. k-means algorithm
            4. Gaussian mixture model with Bayes classifier
        param:
            1. data_for_clustering - pandas DataFrame (10000, 82), where
                values are the mean spendings of customers for every
                category
            2. customer_ids - numpy array (10000, ) with all customer ids
            3. name - string that represents name of embedding algorithm
    """
    # Data embedding
    print("\t0. Transforming data using " + name + " method")
    time_start = time.time()

    data_embedded = \
        dimensionality_reduction.data_embedding(data_for_clustering,
                                                customer_ids, name)

    time_end = (time.time() - time_start) / 60
    print("\t   Data folder: data/output/")
    print("\t   Done. With time " + str(time_end) + " min")

    # Clustering using DBSCAN algorithm
    print("\t1. DBSCAN clustering")
    time_start = time.time()

    clustering.dbscan_clustering(data_for_clustering, name,
                                 auxiliary_data=data_embedded)
    data_for_clustering = data_for_clustering.drop(columns='cluster')

    time_end = (time.time() - time_start) / 60
    print("\t   Images folder: plots/clustering/" + name + "/DBSCAN/")
    print("\t   Done. With time " + str(time_end) + " min")

    # Clustering using OPTICS algorithm
    print("\t2. OPTICS clustering")
    time_start = time.time()

    clustering.optics_clustering(data_for_clustering, name,
                                 auxiliary_data=data_embedded)
    data_for_clustering = data_for_clustering.drop(columns='cluster')

    time_end = (time.time() - time_start) / 60
    print("\t   Images folder: plots/clustering/" + name + "/OPTICS/")
    print("\t   Done. With time " + str(time_end) + " min")

    # Clustering using k-means algorithm with
    # Silhouette method for clusters number choosing
    print("\t3. k-means clustering (Silhouette method)")
    time_start = time.time()

    clustering.kmeans_clustering(data_for_clustering, name,
                                 "k-means with Silhouette",
                                 clusters_number=clustering.silhouette_method(
                                     data_embedded,
                                     "clustering/" + name +
                                     "/k-means with Silhouette"),
                                 auxiliary_data=data_embedded)
    data_for_clustering = data_for_clustering.drop(columns='cluster')

    time_end = (time.time() - time_start) / 60
    print("\t   Images folder: plots/clustering/" + name +
          "/k-means with Silhouette/")
    print("\t   Done. With time " + str(time_end) + " min")

    # Clustering using k-means algorithm with
    # Elbow (Knee) method for clusters number choosing
    print("\t4. k-means clustering (Elbow (Knee) method)")
    time_start = time.time()

    clustering.kmeans_clustering(data_for_clustering, name,
                                 "k-means with Elbow",
                                 clusters_number=clustering.elbow_method(
                                     data_embedded,
                                     "clustering/" + name +
                                     "/k-means with Elbow"),
                                 auxiliary_data=data_embedded)
    data_for_clustering = data_for_clustering.drop(columns='cluster')

    time_end = (time.time() - time_start) / 60
    print("\t   Images folder: plots/clustering/" + name +
          "/k-means with Elbow/")
    print("\t   Done. With time " + str(time_end) + " min")

    # Clustering using Gaussian mixture model with Bayes classifier
    print("\t5. Gaussian mixture clustering")
    time_start = time.time()

    clustering.gaussian_mixture(data_for_clustering, name,
                                auxiliary_data=data_embedded)
    data_for_clustering = data_for_clustering.drop(columns='cluster')

    time_end = (time.time() - time_start) / 60
    print("\t   Images folder: plots/clustering/" + name +
          "/gaussian mixture/")
    print("\t   Done. With time " + str(time_end) + " min")

    return


def main():
    """
        Main function.
    """
    # Data import
    print("========== Importing data ==========")
    time_start = time.time()

    data_for_clustering, data_for_lda, customer_ids, category_names = \
        importing_data()

    time_end = (time.time() - time_start) / 60
    print("Done. With time " + str(time_end) + " min\n")

    # Raw data clustering
    print("========== Initial data clustering ==========")
    time_start = time.time()

    initial_data_clustering(data_for_clustering, data_for_lda, customer_ids,
                            category_names)

    time_end = (time.time() - time_start) / 60
    print("Done. With time " + str(time_end) + " min\n")

    # Clustering data that transformed using MDS algorithm
    print("========== MDS embedded data clustering ==========")
    time_start = time.time()

    embedded_data_clustering(data_for_clustering, customer_ids, "MDS")

    time_end = (time.time() - time_start) / 60
    print("Done. With time " + str(time_end) + " min\n")

    # Clustering data that transformed using t-SNE method
    print("========== t-SNE embedded data clustering ==========")
    time_start = time.time()

    embedded_data_clustering(data_for_clustering, customer_ids, "t-SNE")

    time_end = (time.time() - time_start) / 60
    print("Done. With time " + str(time_end) + " min\n")

    # Clustering data that transformed using Visual Embedding algorithm
    print("========== Spectral Embedding data clustering ==========")
    time_start = time.time()

    embedded_data_clustering(data_for_clustering, customer_ids,
                             "spectral embedding")

    time_end = (time.time() - time_start) / 60
    print("Done. With time " + str(time_end) + " min\n")

    return


if __name__ == "__main__":
    main()
