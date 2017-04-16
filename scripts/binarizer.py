import sys
from datetime import datetime
from multiprocessing import Pool

import numpy as np
import pandas as pd


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


def process_chunk_week(chunk: pd.SparseDataFrame, chunk_number: int, week: int, min_usage: int):
    """ Asynchronously process the input data which is a
        certain week of a certain chunk. Save the result to a pickle file. """

    print('\nProcessing week %d' % week)

    # fix the columns
    print('Removing week level from columns')
    chunk.columns = chunk.columns.droplevel(0)

    # get the dummy columns
    print('Making dummies for hashtags')
    dummies_hashtags = chunk['hashtags'].str.get_dummies(sep=',')
    dummies_hashtags.columns = dummies_hashtags.columns.map(lambda c: 'hashtag_' + c)

    # get rid of dummy columns with usage below min_usage
    print('Filtering hashtag dummies')
    usage = dummies_hashtags.sum(0)
    high_usage = dummies_hashtags[np.where(usage >= min_usage)[0]]
    low_usage = dummies_hashtags[np.where(usage < min_usage)[0]]
    dummies_hashtags = high_usage
    dummies_hashtags['other_hashtags'] = low_usage.sum(1)

    print('There are %d hashtag columns' % dummies_hashtags.shape[1])

    print('Making dummies for mentions')
    dummies_mentions = chunk['mentions'].str.get_dummies(sep=',')
    dummies_mentions.columns = dummies_mentions.columns.map(lambda c: 'mention_' + c)

    print('Filtering mention dummies')
    usage = dummies_mentions.sum(0)
    high_usage = dummies_mentions[np.where(usage >= min_usage)[0]]
    low_usage = dummies_mentions[np.where(usage < min_usage)[0]]
    dummies_mentions = high_usage
    dummies_mentions['other_mentions'] = low_usage.sum(1)

    print('There are %d mention columns' % dummies_mentions.shape[1])

    print('Making dummies for urls')
    dummies_urls = chunk['urls'].str.get_dummies(sep=',')
    dummies_urls.columns = dummies_urls.columns.map(lambda c: 'url_' + c)

    print('Filtering url dummies')
    usage = dummies_urls.sum(0)
    high_usage = dummies_urls[np.where(usage >= min_usage)[0]]
    low_usage = dummies_urls[np.where(usage < min_usage)[0]]
    dummies_urls = high_usage
    dummies_urls['other_urls'] = low_usage.sum(1)

    print('There are %d url columns' % dummies_urls.shape[1])

    # concatenate to one big data frame
    print('Concatenating dummies and copying tweets')
    dummies: pd.SparseDataFrame = pd.concat([dummies_hashtags, dummies_mentions, dummies_urls], axis=1)
    dummies['tweets'] = chunk['tweets']

    # save to a pickle
    print('Saving')
    dummies.to_pickle('../data/chunks/chunk_%d_week_%d.pkl' % (chunk_number, week))


def binarize_dataset():
    """ Binarize the whole dataset. Use the prepared data
        and save the binarized data. Process the data in chunks. """

    print('Binarizing the dataset')

    # constants
    chunk_size = 4000
    total_chunks = 8261630 / chunk_size
    min_usage = 4

    # read the starting chunk
    starting_chunk = 0
    if len(sys.argv) >= 3:
        starting_chunk = int(sys.argv[2])
        print('Resuming from chunk %d' % starting_chunk)

    # read the starting week
    starting_week = 0
    if len(sys.argv) >= 4:
        starting_week = int(sys.argv[3])
        print('Resuming from week %d' % starting_week)

    # make a pool
    pool = Pool()

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

        processes = []

        # process all weeks
        for week in range(first_week, last_week + 1):
            # skip weeks before the starting week
            if week < starting_week:
                print('Skipping week %d' % week)
                continue

            starting_week = 0

            # get just the required data
            chunk: pd.SparseDataFrame = data[[str(week)]]

            # process the data in a worker
            process = pool.apply_async(process_chunk_week, (chunk, i, week, min_usage))
            processes.append(process)

        # wait for the processes before moving to the next chunk
        for process in processes:
            process.wait()



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
