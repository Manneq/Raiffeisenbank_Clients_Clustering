"""
    File 'clustering.py' consists of functions with clustering methods,
        LDA topic modeling and distance, clusters number choosing methods.
"""
import pandas as pd
import numpy as np
import sklearn.cluster
import sklearn.decomposition
import sklearn.model_selection
import sklearn.neighbors
import sklearn.mixture
import sklearn.metrics
import kneed
import math
import plotting


def silhouette_method(data, folder,
                      max_clusters=102):
    """
        Function to find clusters number for k-means using Silhouette method.
        param:
            1. data - pandas DataFrame (10000, 82) or (10000, 3), where
                values are mean spendings of customers for every category
            2. folder - string path to save plot
            3. max_clusters - int number of maximum clusters (102 as default)
        return:
            clusters_number - int value of optimal clusters number
    """
    silhouette_results = []

    # Do k-means clustering with number of clusters from 2 to
    # max_clusters (102) and compute silhouette scores for every clustering
    for clusters_number in range(2, max_clusters):
        kmeans_model = sklearn.cluster.KMeans(n_clusters=clusters_number,
                                              n_jobs=-1).fit(data)

        silhouette_results.append(sklearn.metrics.silhouette_score(
            data, kmeans_model.labels_))

    silhouette_results = np.array(silhouette_results)

    # Get optimal clusters number as index of max score plus 2
    clusters_number = np.argmax(silhouette_results) + 2

    # Plot scores and optimal number of clusters
    plotting.line_plotting([silhouette_results, np.arange(2, max_clusters),
                            clusters_number],
                           ["Clusters number", "Score"], "Silhouette score",
                           folder)

    return clusters_number


def elbow_method(data, folder,
                 max_clusters=102):
    """
        Function to find clusters number for k-means using Elbow method.
        param:
            1. data - pandas DataFrame (10000, 82) or (10000, 3), where
                values are mean spendings of customers for every category
            2. folder - string path to save plot
            3. max_clusters - int number of maximum clusters (102 as default)
        return:
            Int value of optimal clusters number
    """
    elbow_results = []

    # Do k-means clustering with number of clusters from 2 to
    # max_clusters (102) and compute sum of squared distances of samples
    # to their closest cluster centers as scores
    for clusters_number in range(2, max_clusters):
        kmeans_model = sklearn.cluster.KMeans(n_clusters=clusters_number,
                                              n_jobs=-1).fit(data)

        elbow_results.append(kmeans_model.inertia_)

    elbow_results = np.array(elbow_results)

    # Find the elbow (knee) on scores
    knee_loc = kneed.KneeLocator(np.arange(2, max_clusters), elbow_results,
                                 curve="convex", direction="decreasing",
                                 online=False, interp_method="polynomial")

    # Plot scores and optimal number of clusters
    plotting.line_plotting([elbow_results, np.arange(2, max_clusters),
                            knee_loc.knee],
                           ["Clusters number", "Score"], "Elbow score", folder)

    return knee_loc.knee


def dmdbscan_algorithm(data, folder):
    """
        Function to find optimal distance for DBSCAN using DMDBSCAN algorithm.
        param:
            1. data - pandas DataFrame (10000, 82) or (10000, 3), where
                values are mean spendings of customers for every category
            2. folder - string path to save plot
        return:
            Float value of optimal distance
    """
    # Create Nearest Neighbors model to find distance to the
    # first closest neighbor
    nn_model = sklearn.neighbors.NearestNeighbors(n_neighbors=2,
                                                  n_jobs=-1).fit(data)

    # Get and sort distances
    distances, indices = nn_model.kneighbors(data)
    distances = np.sort(distances, axis=0)[:, 1]

    # Find elbow (knee) on distances
    knee_loc = kneed.KneeLocator(distances,
                                 np.arange(len(distances)),
                                 curve="concave", direction="increasing",
                                 online=False, interp_method="polynomial")

    # Plot distances and optimal distance
    plotting.line_plotting([np.arange(len(distances)), distances,
                            knee_loc.knee],
                           ["Distance", ""],
                           "Optimal distance",
                           folder)

    return knee_loc.knee


def dbscan_clustering(data, name,
                      auxiliary_data=None):
    """
        Function for Density-based spatial clustering of applications
            with noise (DBSCAN) algorithm.
        param:
            1. data - pandas DataFrame (10000, 82), where values are mean
                spendings of customers for every category
            2. name - string name of using data
            3. auxiliary_data - pandas DataFrame of data that need to be
                clustered instead of regular DataFrame
    """
    # Decide what data should be using for clustering
    learning_data = data if auxiliary_data is None else auxiliary_data

    # Implement DBSCAN model with distance choose using DMDBSCAN algorithm
    dbscan_model = \
        sklearn.cluster.DBSCAN(
            min_samples=round(math.log(len(learning_data.index))),
            eps=dmdbscan_algorithm(learning_data,
                                   "clustering/" + name +
                                   "/DBSCAN with DMDBSCAN"),
            n_jobs=-1).fit(learning_data)

    # Plot clusters over auxiliary data
    if auxiliary_data is not None:
        plotting.data_plotting(auxiliary_data, "Auxiliary data clustering",
                               "clustering/" + name + "/DBSCAN with DMDBSCAN",
                               color=dbscan_model.labels_)

    # Plot clusters over data
    plotting.data_plotting(data, "DBSCAN clustering",
                           "clustering/" + name + "/DBSCAN with DMDBSCAN",
                           color=dbscan_model.labels_)

    data['cluster'] = dbscan_model.labels_

    # Plot every cluster as a barchart of mean spendings
    for i in range(len(np.unique(dbscan_model.labels_)) - 1):
        cluster_data = data[data['cluster'] == i].drop(columns='cluster')
        mean_cluster_data = cluster_data.mean(axis=0).sort_values()

        plotting.bar_plotting([np.array(mean_cluster_data.values),
                               np.array(mean_cluster_data.index)],
                              ["Category", "Mean spendings"],
                              "Cluster " + str(i) + " (" +
                              str(len(cluster_data.index)) + " customers)",
                              "clustering/" + name +
                              "/DBSCAN with DMDBSCAN/clusters")

    return


def optics_clustering(data, name,
                      auxiliary_data=None):
    """
        Function for Oredering points to identify the clustering structure
            (OPTICS) algorithm.
        param:
            1. data - pandas DataFrame (10000, 82), where values are mean
                spendings of customers for every category
            2. name - string name of using data
            3. auxiliary_data - pandas DataFrame of data that need to be
                    clustered instead of regular DataFrame
    """
    # Decide what data should be using for clustering
    learning_data = data if auxiliary_data is None else auxiliary_data

    # Implement OPTICS model
    optics_model = \
        sklearn.cluster.OPTICS(
            min_samples=round(math.log(len(learning_data.index))),
            n_jobs=-1).fit(learning_data)
    # Extract OPTICS clusters using DBSCAN algorithm with distance choosing
    # as DMDBSCAN algorithm
    labels = sklearn.cluster.cluster_optics_dbscan(
        reachability=optics_model.reachability_,
        core_distances=optics_model.core_distances_,
        ordering=optics_model.ordering_,
        eps=dmdbscan_algorithm(learning_data,
                               "clustering/" + name + "/OPTICS"))

    # Plot clusters over auxiliary data
    if auxiliary_data is not None:
        plotting.data_plotting(auxiliary_data, "Auxiliary data clustering",
                               "clustering/" + name + "/OPTICS",
                               color=labels)

    # Plot clusters over data
    plotting.data_plotting(data, "OPTICS clustering",
                           "clustering/" + name + "/OPTICS",
                           color=labels)

    data['cluster'] = labels

    # Plot every cluster as a barchart of mean spendings
    for i in range(len(np.unique(labels)) - 1):
        cluster_data = data[data['cluster'] == i].drop(columns='cluster')
        mean_cluster_data = cluster_data.mean(axis=0).sort_values()

        plotting.bar_plotting([np.array(mean_cluster_data.values),
                               np.array(mean_cluster_data.index)],
                              ["Category", "Mean spendings"],
                              "Cluster " + str(i) + " (" +
                              str(len(cluster_data.index)) + " customers)",
                              "clustering/" + name + "/OPTICS/clusters")

    return


def kmeans_clustering(data, name, clustering_name,
                      clusters_number=8, auxiliary_data=None):
    """
        Function for k-means algorithm.
        param:
            1. data - pandas DataFrame (10000, 82), where values are mean
                spendings of customers for every category
            2. name - string name of using data
            3. clustering_name - string more precise name of clustering
            4. clusters_number - int number of clusters (8 as default)
            5. auxiliary_data - pandas DataFrame of data that need to be
                clustered instead of regular DataFrame
    """
    # Decide what data should be using for clustering
    learning_data = data if auxiliary_data is None else auxiliary_data

    # Implement k-means model
    kmeans_model = \
        sklearn.cluster.KMeans(n_clusters=clusters_number, n_jobs=-1).\
        fit(learning_data)

    # Plot clusters over auxiliary data
    if auxiliary_data is not None:
        plotting.data_plotting(auxiliary_data, "Auxiliary data clustering",
                               "clustering/" + name + "/" + clustering_name,
                               color=kmeans_model.labels_)

    # Plot clusters over data
    plotting.data_plotting(data, clustering_name,
                           "clustering/" + name + "/" + clustering_name,
                           color=kmeans_model.labels_)

    data['cluster'] = kmeans_model.labels_

    # Plot every cluster as a barchart of mean spendings
    for i in range(len(np.unique(kmeans_model.labels_))):
        cluster_data = data[data['cluster'] == i].drop(columns='cluster')
        mean_cluster_data = cluster_data.mean(axis=0).sort_values()

        plotting.bar_plotting([np.array(mean_cluster_data.values),
                               np.array(mean_cluster_data.index)],
                              ["Category", "Mean spendings"],
                              "Cluster " + str(i) + " (" +
                              str(len(cluster_data.index)) + " customers)",
                              "clustering/" + name + "/"
                              + clustering_name + "/clusters")

    return


def topic_clustering(data, lda_results, topics_number):
    """
        Function for clustering data using dominant topics from
            LDA topic modeling results.
        param:
            1. data - pandas DataFrame (10000, 82), where values are mean
                spendings of customers for every category
            2. auxiliary_data - pandas DataFrame of LDA topic modeling results
            3. topics_number - int number of topics
    """
    # Extract topic labels as cluster labels
    topic_labels = lda_results[['dominant_topic']].values.\
        reshape((len(lda_results[['dominant_topic']].values), ))

    # Plot clusters over LDA results
    plotting.data_plotting(lda_results.drop(columns=['dominant_topic']),
                           "Auxiliary data clustering",
                           "clustering/initial data/clustering by LDA topics",
                           color=topic_labels)

    # Plot clusters over data
    plotting.data_plotting(data, "Clustering by LDA topics",
                           "clustering/initial data/clustering by LDA topics",
                           color=topic_labels)

    data['dominant_topic'] = topic_labels

    # Plot every cluster as a barchart of mean spendings
    for i in range(topics_number):
        cluster_data = data[data['dominant_topic'] == i].\
            drop(columns='dominant_topic')
        mean_cluster_data = cluster_data.mean(axis=0).sort_values()

        plotting.bar_plotting([np.array(mean_cluster_data.values),
                               np.array(mean_cluster_data.index)],
                              ["Category", "Mean spendings"],
                              "Cluster " + str(i) + " (" +
                              str(len(cluster_data.index)) + " customers)",
                              "clustering/initial data/"
                              "clustering by LDA topics/clusters")

    return


def gaussian_mixture(data, name,
                     auxiliary_data=None):
    """
        Function clustering using Gaussian mixture model with Bayes classifier.
        param:
            1. data - pandas DataFrame (10000, 82), where values are mean
                spendings of customers for every category
            2. name - string name of using data
            3. auxiliary_data - pandas DataFrame of data that need to be
                clustered instead of regular DataFrame
    """
    # Decide what data should be using for clustering
    learning_data = data if auxiliary_data is None else auxiliary_data

    # Implement Gaussian mixture model
    bgm_model = sklearn.mixture.\
        BayesianGaussianMixture(n_components=5, max_iter=1000,
                                weight_concentration_prior=1e+03).\
        fit(learning_data)
    labels = bgm_model.predict(learning_data)

    # Plot clusters over auxiliary data
    if auxiliary_data is not None:
        plotting.data_plotting(auxiliary_data, "Auxiliary data clustering",
                               "clustering/" + name + "/gaussian mixture",
                               color=labels)

    # Plot clusters over data
    plotting.data_plotting(data, "Gaussian mixture clustering",
                           "clustering/" + name + "/gaussian mixture",
                           color=labels)

    data['cluster'] = labels

    # Plot every cluster as a barchart of mean spendings
    for i in range(len(np.unique(labels))):
        cluster_data = data[data['cluster'] == i].drop(columns='cluster')
        mean_cluster_data = cluster_data.mean(axis=0).sort_values()

        plotting.bar_plotting([np.array(mean_cluster_data.values),
                               np.array(mean_cluster_data.index)],
                              ["Category", "Mean spendings"],
                              "Cluster " + str(i) + " (" +
                              str(len(cluster_data.index)) + " customers)",
                              "clustering/" + name +
                              "/gaussian mixture/clusters")

    return


def lda_performing(data_for_lda, customer_ids, category_names):
    """
        Function for topic modeling using the Latent Dirichlet Algorithm (LDA).
        param:
            1. data_for_lda - pandas DataFrame (10000, 82), where values are
                the transactions number of customers for every category
            2. customer_ids - numpy array (10000, ) with all customer ids
            3. category_names - numpy array (82, ) with names of categories
        return:
            1. lda_results - pandas DataFrame results of LDA modeleing
            2. Int value of topics number
    """
    # Create LDA with standard parameters
    lda_model = \
        sklearn.decomposition.LatentDirichletAllocation(n_jobs=-1)

    # Do a grid search over parameters of LDA
    search_params = {'n_components': [5, 10, 15, 20, 25, 30],
                     'learning_decay': [.5, .7, .9],
                     'max_iter': [10, 15, 20, 25, 30]}
    # Create grid search model
    grid_search_model = \
        sklearn.model_selection.GridSearchCV(lda_model,
                                             param_grid=search_params,
                                             n_jobs=-1)
    grid_search_model.fit(data_for_lda)

    # Get the LDA model with the best score
    lda_model = grid_search_model.best_estimator_

    # Fit and transform model
    lda_results = lda_model.fit_transform(data_for_lda)

    # Create topic names
    topic_names = \
        np.array(["topic " + str(i) for i in
                  range(lda_results.shape[1])])

    # Create and save DataFrame of results
    lda_results = pd.DataFrame(data=lda_results, columns=topic_names,
                               index=customer_ids)
    lda_results.index.name = "customer_id"
    dominant_topics = np.argmax(lda_results.values, axis=1)
    lda_results['dominant_topic'] = dominant_topics
    lda_results.to_csv(path_or_buf="data/output/lda_results.csv")

    # Plot probability of topics for every customer as a barchart
    for customer_id in customer_ids:
        customer_data = \
            lda_results.drop(columns=['dominant_topic']).loc[[customer_id]].\
            sort_values(by=customer_id, axis=1)

        plotting.bar_plotting([np.array(customer_data.values[0]),
                               np.array(customer_data.columns)],
                              ["Category", "Probability"],
                              "Customer " + customer_id,
                              "LDA/customers",
                              color='m')

    # Create DataFrame of topics distribution over customers
    topic_distribution = pd.DataFrame()
    topic_distribution['customers_number'] = \
        lda_results['dominant_topic'].value_counts(sort=False).\
        reindex(np.arange(len(topic_names)), fill_value=0)
    topic_distribution.index = \
        lda_results.columns[:len(topic_names)]
    topic_distribution.index.name = "topic"
    topic_distribution = \
        topic_distribution.sort_values(by=['customers_number'], axis=0)

    # plot the distribution
    plotting.bar_plotting([np.array(topic_distribution['customers_number'].
                                    values),
                           np.array(topic_distribution.index)],
                          ["Topic", "Number of customers"],
                          "Topics distribution over customers",
                          "LDA")

    # Create DataFrame for topics
    topics_data = \
        pd.DataFrame(data=lda_model.components_ / lda_model.components_.
                     sum(axis=1)[:, np.newaxis],
                     columns=data_for_lda.columns,
                     index=topic_names)
    topics_data.index.name = "topic"

    # Plot probability of categories for every topic as a barchart
    for topic_name in topic_names:
        topic_data = \
            topics_data.loc[[topic_name]].sort_values(by=topic_name, axis=1)

        plotting.bar_plotting([np.array(topic_data.values[0]),
                               np.array(topic_data.columns)],
                              ["Category", "Probability"],
                              topic_name,
                              "LDA/topics")

    # Plot probability of topics for every category as a barchart
    for i in range(len(category_names)):
        category_data = topics_data.iloc[:, i].sort_values(axis=0)

        plotting.bar_plotting([np.array(category_data.values),
                               np.array(category_data.index)],
                              ["Topic", "Probability"],
                              "Category " + category_names[i],
                              "LDA/categories",
                              color='g')

    return lda_results, len(topic_names)
