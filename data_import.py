"""
    File 'data_import.py' used to load, preprocess, vectorize dataset
        and to find distributions across it.
"""
import pandas as pd
import numpy as np
import plotting


def loading_data():
    """
        Method to load initial Raiffeisenbank clients transactions dataset.
        return:
            data - pandas DataFrame of initial Raiffeisenbank clients
                transactions dataset
    """
    data = pd.read_csv("data/input/train_set.csv", delimiter=',',
                       index_col=False, dtype='unicode')

    return data


def preprocessing_data():
    """
        Method to preprocess and mapping data.
        return:
            1. data - pandas DataFrame of mapped data
            2. Numpy array (10000, ) with all customer ids
            3. Numpy array (82, ) with names of categories
    """
    # Data loading and picking only 3 feature that can be clustered
    data = loading_data()[['customer_id', 'amount', 'mcc']]

    # Loading of mapping category codes to mcc
    data_mapping = pd.read_csv("data/input/mcc2big.csv", delimiter=',',
                               index_col=False, dtype='unicode')
    data = pd.merge(data, data_mapping, on='mcc')

    # Loading of mapping category names on category codes
    data_mapping = pd.read_csv("data/input/87_categories.csv", delimiter=',',
                               index_col=False, dtype='unicode')
    data = pd.merge(data, data_mapping, on='category')

    # Dropping category codes and mcc
    data = data.drop(columns=['category', 'mcc'])

    return data, data['customer_id'].unique(), data['category_name'].unique()


def data_distributions(data, customer_ids, category_names):
    """
        Method to find and plot distributions across the dataset:
            1. Customers annual spendings
            2. Customers spendings
            3. Category transactions
            4. category average spendings
        param:
            1. data - pandas DataFrame of mapped data
            2. customer_ids - numpy array (10000, ) with all customer ids
            3. category_names - numpy array (82, ) with names of categories
    """
    annual_spendings, spendings, category_spendings = [], [], []
    category_numbers = np.zeros((len(category_names), ))

    # Find customers distributions
    for customer_id in customer_ids:
        customer_data = data[data['customer_id'] == customer_id]
        annual_spendings.append(pd.to_numeric(customer_data['amount']).sum())

        for i in range(len(customer_data.index)):
            spendings.append(pd.to_numeric(customer_data['amount'].values[i]))

    # Find categories distributions
    for i in range(len(category_names)):
        category_data = data[data['category_name'] == category_names[i]]
        category_numbers[i] += len(category_data.index)
        category_spendings.append(
            pd.to_numeric(category_data['amount']).mean())

    # Plot customers distributions
    plotting.hist_plotting(np.array(annual_spendings),
                           ["Annual spendings", "Frequency"],
                           "Customers annual spendings distribution")
    plotting.hist_plotting(np.array(spendings),
                           ["Spendings", "Frequency"],
                           "Customers spendings distribution")

    # Sort and plot category distributions
    category_numbers = np.array(category_numbers)
    sorted_indexes = np.argsort(category_numbers)

    plotting.bar_plotting([np.take_along_axis(category_numbers,
                                              sorted_indexes, axis=0),
                           np.take_along_axis(category_names,
                                              sorted_indexes, axis=0)],
                          ["Category", "Number of transactions"],
                          "Category transactions",
                          "data distributions")

    category_spendings = np.array(category_spendings)
    sorted_indexes = np.argsort(category_spendings)

    plotting.bar_plotting([np.take_along_axis(category_spendings,
                                              sorted_indexes, axis=0),
                           np.take_along_axis(category_names,
                                              sorted_indexes, axis=0)],
                          ["Category", "Average spendings"],
                          "Category average spendings",
                          "data distributions")

    return


def data_vectorization(data, customer_ids, category_names):
    """
        Method to vectorize dataset for topic modeling and clustering.
        param:
            1. data - pandas DataFrame of mapped data
            2. customer_ids - numpy array (10000, ) with all customer ids
            3. category_names - numpy array (82, ) with names of categories
        return:
            1. data_for_clustering - pandas DataFrame (10000, 82), where
                values are mean spendings of customers for every category
            2. data_for_lda - pandas DataFrame (10000, 82), where values are
                the transactions number of customers for every category
    """
    data_for_clustering = np.zeros((len(customer_ids), len(category_names)))
    data_for_lda = np.zeros((len(customer_ids), len(category_names)))

    # Compute customers mean spendings and number of transactions
    # for every category
    for i in range(len(customer_ids)):
        temp_id_data = data[data['customer_id'] == customer_ids[i]]
        for j in range(len(category_names)):
            temp_cat_data = temp_id_data[temp_id_data['category_name'] ==
                                         category_names[j]]
            data_for_clustering[i][j] = \
                pd.to_numeric(temp_cat_data['amount']).mean()
            data_for_lda[i][j] = temp_cat_data['amount'].count()

    # Create and save to csv DataFrame for clustering
    data_for_clustering = pd.DataFrame(data_for_clustering,
                                       columns=category_names,
                                       index=customer_ids)
    data_for_clustering.index.name = "customer_id"
    data_for_clustering = data_for_clustering.fillna(value=0)
    data_for_clustering.to_csv(
        path_or_buf="data/output/data_for_clustering.csv")

    # Create and save to csv DataFrame for topic modeling
    data_for_lda = pd.DataFrame(data_for_lda, columns=category_names,
                                index=customer_ids)
    data_for_lda.index.name = "customer_id"
    data_for_lda = data_for_lda.fillna(value=0)
    data_for_lda.to_csv(path_or_buf="data/output/data_for_lda.csv")

    return data_for_clustering, data_for_lda
