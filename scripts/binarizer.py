import sys
import pandas as pd
from datetime import datetime
from multiprocessing import Pool
from os import listdir
from parse import parse


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

    print('[%d] Processing week %d' % (week, week))

    # fix the columns
    print('[%d] Removing week level from columns' % week)
    chunk.columns = chunk.columns.droplevel(0)

    # get the dummy columns for hashtags
    print('[%d] Making dummies for hashtags' % week)
    dummies_hashtags: pd.SparseDataFrame = chunk['hashtags'].apply(
        lambda v: v.lower() if type(v) == str else '').str.get_dummies(sep=',')
    dummies_hashtags_values = dummies_hashtags.values

    usage = dummies_hashtags_values.sum(0)
    high_usage = usage >= min_usage
    other = dummies_hashtags_values[:, usage < min_usage]
    dummies_hashtags = pd.SparseDataFrame(dummies_hashtags_values[:, high_usage], dummies_hashtags.index,
                                          dummies_hashtags.columns[high_usage].map(lambda c: 'hashtag_' + c))
    dummies_hashtags['other_hashtags'] = other.sum(1)

    print('[%d] There are %d hashtag columns' % (week, dummies_hashtags.shape[1]))

    # get the dummy columns for mentions
    print('[%d] Making dummies for mentions' % week)
    dummies_mentions: pd.SparseDataFrame = chunk['mentions'].apply(
        lambda v: v.lower() if type(v) == str else '').str.get_dummies(sep=',')
    dummies_mentions_values = dummies_mentions.values

    usage = dummies_mentions_values.sum(0)
    high_usage = usage >= min_usage
    other = dummies_mentions_values[:, usage < min_usage]
    dummies_mentions = pd.SparseDataFrame(dummies_mentions_values[:, high_usage], dummies_mentions.index,
                                          dummies_mentions.columns[high_usage].map(lambda c: 'mention_' + c))
    dummies_mentions['other_mentions'] = other.sum(1)

    print('[%d] There are %d mention columns' % (week, dummies_mentions.shape[1]))

    # get the dummy columns for urls
    print('[%d] Making dummies for urls' % week)
    dummies_urls: pd.SparseDataFrame = chunk['urls'].apply(
        lambda v: v.lower() if type(v) == str else '').str.get_dummies(sep=',')
    dummies_urls_values = dummies_urls.values

    usage = dummies_urls_values.sum(0)
    high_usage = usage >= min_usage
    other = dummies_urls_values[:, usage < min_usage]
    dummies_urls = pd.SparseDataFrame(dummies_urls_values[:, high_usage], dummies_urls.index,
                                      dummies_urls.columns[high_usage].map(lambda c: 'url_' + c))
    dummies_urls['other_urls'] = other.sum(1)

    print('[%d] There are %d url columns' % (week, dummies_urls.shape[1]))

    # concatenate to one big data frame
    print('[%d] Concatenating dummies and copying tweets' % week)
    dummies: pd.SparseDataFrame = pd.concat([dummies_hashtags, dummies_mentions, dummies_urls], axis=1)
    dummies['tweets'] = chunk['tweets']

    # save to a pickle
    print('[%d] Saving' % week)
    dummies.to_pickle('../data/chunks/chunk_%d_week_%d.pkl' % (chunk_number, week))


def binarize_dataset():
    """ Binarize the whole dataset. Use the prepared data
        and save the binarized data. Process the data in chunks. """

    print('Binarizing the dataset')

    # constants
    chunk_size = 2000
    total_chunks = 8261630 / chunk_size
    min_usage = 10

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
            process.get()


def merge_chunks():
    """ Merge all created chunks to a single data frame. """

    # find the last chunk generated
    files = listdir('../data/chunks/')
    files = list(map(lambda name: parse('chunk_{:d}_week_{:d}.pkl', name).fixed, files))

    chunks = sorted(files, key=lambda cw: cw[0])
    last_chunk = chunks[-1][0]

    # make sure the last chunk has all weeks
    while True:
        print('Checking weeks of chunk %d' % last_chunk)

        chunk_weeks = list(map(lambda c: c[1], filter(lambda c: c[0] == last_chunk, chunks)))

        # check that all weeks from 23 to 36 are present
        has_all_weeks = True
        for week in range(23, 37):
            if week not in chunk_weeks:
                print('Missing week %d' % week)
                has_all_weeks = False
                break

        if has_all_weeks:
            break
        else:
            last_chunk -= 1

    print('Merging chunks from 0 to %d' % last_chunk)

    data_frames = []

    # start loading the data and merging
    for chunk_number in range(last_chunk + 1):
        print('Adding chunk %d' % chunk_number)

        weeks = []

        # load weeks and merge them
        for week in range(23, 37):
            week_data = pd.read_pickle('../data/chunks/chunk_%d_week_%d.pkl' % (chunk_number, week))
            weeks.append(week_data)

        chunk = pd.concat(weeks, axis=1, keys=list(range(23, 37)))

        # append to the data
        data_frames.append(chunk)

    # save the data
    print('Concatenating chunks')
    data: pd.SparseDataFrame = pd.concat(data_frames)
    print('Saving the data with size %d by %d' % data.shape)
    print('The type of the data is ' + str(type(data)))

    data.fillna(0, inplace=True)
    data.to_pickle('../data/binarized_data.pkl')


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

    elif sys.argv[1] == 'merge':
        merge_chunks()

    else:
        print('Unknown action: ' + sys.argv[1])
