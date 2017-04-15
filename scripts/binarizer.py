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
        pivot[c] = pivot[c].applymap(lambda s: ','.join(s) if type(s) == list else None)

    print('The shape of the pivot table is %d by %d' % pivot.shape)

    # save to a CSV
    print('Saving as pivot_data.csv')
    pivot.to_csv('../data/pivot_data.csv', index=True, index_label='user')


def binarize_dataset():
    """ Binarize the whole dataset. Use the prepared data
        and save the binarized data. Process the data in chunks. """

    print('Binarizing the dataset')

    chunk_size = 10000
    total_chunks = 8261630 / chunk_size

    # read the starting chunk
    starting_chunk = 0
    if len(sys.argv) >= 3:
        starting_chunk = int(sys.argv[2])
        print('Resuming from chunk %d' % starting_chunk)

    # start with loading
    print('Loading the pivot_data.csv')
    types = {'tweets': int, 'hashtags': str, 'mentions': str, 'urls': str}
    data_chunks = pd.read_csv('../data/pivot_data.csv', header=[0, 1], index_col=0, chunksize=chunk_size, dtype=types)

    # process all chunks
    for i, data in enumerate(data_chunks):
        # skip chunks before the starting chunk
        if i < starting_chunk:
            print('Skipping chunk %d' % i)
            continue

        print('\nPROCESSING CHUNK %d of %d\n' % (i, total_chunks))

        print('Converting to sparse data frame')
        data: pd.SparseDataFrame = data.to_sparse(0)

        # swap levels so the week number is the first
        print('Swapping levels')
        data = data.swaplevel(axis=1).sort_index(1)

        # store the minimum and the maximum week number for iterating later
        first_week = int(data.columns.levels[0].min())
        last_week = int(data.columns.levels[0].max())

        # process all weeks
        for week in range(first_week, last_week + 1):
            print('\nProcessing week %d' % week)
            chunk: pd.SparseDataFrame = data[[str(week)]]

            # fix the columns
            print('Removing week level from columns')
            chunk.columns = chunk.columns.droplevel(0)

            # get the dummy columns
            print('Making dummies for hashtags')
            dummies_hashtags = chunk['hashtags'].str.get_dummies(sep=',')
            dummies_hashtags.columns = dummies_hashtags.columns.map(lambda c: 'hashtag_' + c)

            print('Making dummies for mentions')
            dummies_mentions = chunk['mentions'].str.get_dummies(sep=',')
            dummies_mentions.columns = dummies_mentions.columns.map(lambda c: 'mention_' + c)

            print('Making dummies for urls')
            dummies_urls = chunk['urls'].str.get_dummies(sep=',')
            dummies_urls.columns = dummies_urls.columns.map(lambda c: 'url_' + c)

            # concatenate to one big data frame
            print('Concatenating dummies and copying tweets')
            dummies: pd.SparseDataFrame = pd.concat([dummies_hashtags, dummies_mentions, dummies_urls], axis=1)
            dummies['tweets'] = chunk['tweets']

            # save to a pickle
            print('Saving')
            dummies.to_pickle('../data/chunks/chunk_%d_week_%d.pkl' % (i, week))


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
