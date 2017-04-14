import numpy as np
import pandas as pd
import sys
from datetime import datetime
from sklearn.preprocessing import MultiLabelBinarizer


def prepare_dataset():
    """ Pre-process the raw dataset and prepare it
        for binarizing. Loads data.csv and saves the result
        to data_prepared.pkl. """

    print('Preparing the dataset')

    print('Loading data.csv')
    data: pd.DataFrame = pd.read_csv('../data/data.csv')

    # convert timestamps to week numbers
    print('Converting weeks')
    data['week'] = data['week'].apply(lambda w: datetime.strptime(w, '%Y-%m-%d 00:00:00').isocalendar()[1])

    # drop tweets from 43th week
    data: pd.DataFrame = data[data['week'] < 40]

    # drop unused columns
    data.drop(['total_length', 'total_words'], axis=1, inplace=True)

    # process categorical columns
    for c in ['hashtags', 'mentions', 'urls']:
        print('Processing lists in ' + c)
        data[c] = data[c].apply(lambda s: s[1:-1])

    print('The shape of the data is %d by %d' % data.shape)

    # save to a CSV
    print('Saving as prepared_data.csv')
    data.to_csv('../data/prepared_data.csv', index=False)


def pivot_dataset():
    """ Make a pivot table from the dataset. """

    print('Pivoting the dataset')

    # load the data
    print('Loading prepared_data.csv')
    data: pd.DataFrame = pd.read_csv('../data/prepared_data.csv')

    # process categorical columns - split to lists
    for c in ['hashtags', 'mentions', 'urls']:
        print('Processing lists in ' + c)
        data[c] = data[c].apply(lambda s: s.split(',') if type(s) == str else [])

    # make the pivot table
    print('Making the pivot table')
    pivot: pd.DataFrame = data.set_index(['user', 'week']).unstack('week')
    del data

    # fill missing tweets
    print('Filling NAs in tweets')
    pivot['tweets'] = pivot['tweets'].fillna(0)

    # process categorical columns again - join back to strings
    for c in ['hashtags', 'mentions', 'urls']:
        print('Processing lists in ' + c)
        pivot[c] = pivot[c].apply(lambda s: ','.join(s) if type(s) == list else '')

    print('The shape of the pivot table is %d by %d' % pivot.shape)

    # save to a CSV
    print('Saving as pivot_data.csv')
    pivot.to_csv('../data/pivot_data.csv', index=True, index_label='user')


def binarize_dataset():
    """ Binarize the whole dataset. Use the prepared data
        and save the binarized data. """

    pass


def binarize(data: pd.SparseDataFrame, column: str):
    """ Binarize a certain column. Save the new columns
        back to the provided data frame, in place. """

    # use the label binarizer to get the new columns
    bin_columns = MultiLabelBinarizer(sparse_output=True).fit_transform(data[column].apply(lambda x: x.split('|')))

    # filter only most used values
    usage = bin_columns.sum(0)
    columns_with_high_usage = bin_columns[:, np.where(usage >= 200)[0]]
    columns_with_low_usage = bin_columns[:, np.where(usage < 200)[0]]
    has_other = columns_with_low_usage.sum(1).astype(bool).astype(int)

    # insert back to the dataset
    num_columns = columns_with_high_usage.shape[1]

    data.drop(column, axis=1, inplace=True)
    data[column + '_other'] = pd.SparseSeries(np.asarray(has_other).ravel(), fill_value=0)

    for i in range(num_columns):
        data[column + '_' + str(i)] = pd.SparseSeries(columns_with_high_usage[:, i].toarray().ravel(), fill_value=0)


if __name__ == '__main__':
    if len(sys.argv) == 1:
        # by default, binarize the dataset
        binarize_dataset()

    elif sys.argv[1] == 'binarize':
        binarize_dataset()

    elif sys.argv[1] == 'prepare':
        prepare_dataset()

    elif sys.argv[1] == 'pivot':
        pivot_dataset()

    else:
        print('Unknown action: ' + sys.argv[1])
