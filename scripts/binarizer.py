import numpy as np
import pandas as pd
import sys
from sklearn.preprocessing import MultiLabelBinarizer


def binarize(data: pd.SparseDataFrame, column: str):
    """ Binarize a certain column. """

    print('\n\nBinarizing column ' + column)

    # use the label binarizer to get the new columns
    print('Using MultiLabelBinarizer')
    bin_columns = MultiLabelBinarizer(sparse_output=True).fit_transform(data[column].apply(lambda x: x.split('|')))

    # filter only most used values
    print('Filtering columns')
    usage = bin_columns.sum(0)
    columns_with_high_usage = bin_columns[:, np.where(usage >= 200)[0]]
    columns_with_low_usage = bin_columns[:, np.where(usage < 200)[0]]
    has_other = columns_with_low_usage.sum(1).astype(bool).astype(int)

    # insert back to the dataset
    num_columns = columns_with_high_usage.shape[1]
    print('Inserting ' + str(num_columns + 1) + ' columns')

    print('Inserting other column')
    data.drop(column, axis=1, inplace=True)
    data[column + '_other'] = pd.SparseSeries(np.asarray(has_other).ravel(), fill_value=0)

    for i in range(num_columns):
        print('Inserting column', i, '/', num_columns)
        data[column + '_' + str(i)] = pd.SparseSeries(columns_with_high_usage[:, i].toarray().ravel(), fill_value=0)


def binarize_hashtags():
    """ Convert hashtags column to N binary columns. """

    print('Loading hashtags')
    data = pd.read_csv('../data/hashtags_for_binarizer.csv').to_sparse(fill_value=0)
    binarize(data, 'hashtags')
    print('Saving hashtags')
    data.to_pickle('../data/hashtags_binarized.pkl')


def binarize_mentions():
    """ Convert mentions column to N binary columns. """

    print('Loading mentions')
    data = pd.read_csv('../data/mentions_for_binarizer.csv').to_sparse(fill_value=0)
    binarize(data, 'mentions')
    print('Saving mentions')
    data.to_pickle('../data/mentions_binarized.pkl')


def binarize_urls():
    """ Convert urls column to N binary columns. """

    print('Loading urls')
    data = pd.read_csv('../data/urls_for_binarizer.csv').to_sparse(fill_value=0)
    binarize(data, 'urls')
    print('Saving urls')
    data.to_pickle('../data/urls_binarized.pkl')


def binarize_dataset():
    """ Load the dataset, prepare it, and convert categorical
        columns to N columns with 1 and 0. This script presumes
        there are already separate CSVs for hashtags, mentions, and URLs. """

    binarize_hashtags()
    binarize_mentions()
    binarize_urls()


if __name__ == '__main__':
    if len(sys.argv) == 1:
        binarize_dataset()
    elif sys.argv[1] == 'hashtags':
        binarize_hashtags()
    elif sys.argv[1] == 'mentions':
        binarize_mentions()
    elif sys.argv[1] == 'urls':
        binarize_urls()
