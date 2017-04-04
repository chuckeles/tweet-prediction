"""
This file contains helpers for the Jupyter notebooks. There are
common things like processing a data frame, splitting it into the train
and test sets, etc.
"""

import numpy as np
import pandas as pd


def load_pivot_numbers():
    """ Load the dataset with numbers only and make a pivot table. """

    data = pd.read_csv('../data/data_numbers_only.csv')
    return data.pivot_table(index='user', columns='week', values=['tweets', 'hashtags', 'mentions', 'urls'],
                            aggfunc=np.sum, fill_value=0)


def make_target(data, target_week):
    """ Create the target column. """

    data['target'] = data['tweets'][target_week] > 0
    return data.drop(target_week, axis=1, level=1)


def balance_data(data):
    """ Make a balanced dataset. Active and inactive will have the same count. """

    active = data[data['target'] == True]
    inactive = data[data['target'] == False]

    if active.shape[0] > inactive.shape[0]:
        active = active.sample(inactive.shape[0])
    else:
        inactive = inactive.sample(active.shape[0])

    return pd.concat([active, inactive])


def normalize_data(data):
    """ Normalize a dataset, divide all cells by column sums. """

    return data.div(data.sum()).fillna(0)


def apply_time_decay(data, first_week, target_week):
    """ Apply a time decay effect on all columns. """

    decay = data

    for week in range(first_week, target_week):
        decay.loc[:, ('tweets', week)] = decay['tweets'][week] / np.sqrt(target_week - week)
        decay.loc[:, ('hashtags', week)] = decay['hashtags'][week] / np.sqrt(target_week - week)
        decay.loc[:, ('mentions', week)] = decay['mentions'][week] / np.sqrt(target_week - week)
        decay.loc[:, ('urls', week)] = decay['urls'][week] / np.sqrt(target_week - week)

    return decay


def split_train_test(data, ratio=.7):
    """ Split the data into a train and a test dataset. Ratio defines how much will be the test part. """

    train_rows = np.random.rand(data.shape[0]) < ratio
    train = data[train_rows].drop('target', axis=1)
    train_target = data[train_rows]['target']
    test = data[~train_rows].drop('target', axis=1)
    test_target = data[~train_rows]['target']

    return train, train_target, test, test_target
