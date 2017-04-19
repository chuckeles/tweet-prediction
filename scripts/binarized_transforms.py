"""
This module has custom Scikit-learn transformers for making
pipelines and doing grid search. The difference between these and the originals
is that these work with the binarized dataset which has different columns.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class TargetMaker(BaseEstimator, TransformerMixin):
    def __init__(self, target_week):
        self.target_week = target_week

    def fit(self, data, target=None):
        return self

    def transform(self, data):
        data['target'] = data[(self.target_week, 'tweets')] > 0
        data.drop(self.target_week, axis=1, inplace=True)

        return data


class ClassBalancer(BaseEstimator, TransformerMixin):
    """ Balances the dataset so both classes have the same amount. """

    def __init__(self):
        self.activeIndices = []
        self.inactiveIndices = []

    def fit(self, data, target):
        # split classes
        active = data[target]
        inactive = data[~target]

        # sample
        if active.shape[0] > inactive.shape[0]:
            active = active.sample(inactive.shape[0])
        else:
            inactive = inactive.sample(active.shape[0])

        # store indices
        self.activeIndices = active.index
        self.inactiveIndices = inactive.index

        return self

    def transform(self, data):
        # take only stored indices
        active = data.loc[self.activeIndices]
        inactive = data.loc[self.inactiveIndices]

        return pd.concat([active, inactive])


class Normalizer(BaseEstimator, TransformerMixin):
    """ Normalizes the dataset so the sums per week are 1. Only normalizes
        columns that contain actual counts and ignores the binary columns. """

    def __init__(self, ignore_binarized_columns=True, verbose=False):
        self.ignore_binarized_columns = ignore_binarized_columns
        self.verbose = verbose

    def fit(self, data, target=None):
        return self

    def transform(self, data):
        for week in data.columns.get_level_values(0):
            columns_to_process = ['tweets', 'other_hashtags', 'other_mentions', 'other_urls'] if \
                self.ignore_binarized_columns else \
                data.loc[:, [week]].columns.get_level_values(1)

            for column in columns_to_process:
                if self.verbose:
                    print('Normalizing column', week, column)

                column_sum = data[(week, column)].sum()
                if column_sum > 0:
                    data[(week, column)] = data[(week, column)].div(column_sum).fillna(0)

        return data


class TimeDecayApplier(BaseEstimator, TransformerMixin):
    """ Apply a time decay on the data. Weeks that occurred
        further before the target will have less power. Ignore categorical columns. """

    def __init__(self, target_week, ignore_binarized_columns=True, verbose=False):
        self.target_week = target_week
        self.ignore_binarized_columns = ignore_binarized_columns
        self.verbose = verbose

    def fit(self, data, target=None):
        return self

    def transform(self, data):
        for week in data.columns.get_level_values(0):
            columns_to_process = ['tweets', 'other_hashtags', 'other_mentions', 'other_urls'] if \
                self.ignore_binarized_columns else \
                data.loc[:, [week]].columns.get_level_values(1)

            time_decay = np.sqrt(max(1, self.target_week - week))

            for column in columns_to_process:
                if self.verbose:
                    print('Applying time-decay to column', week, column)

                data[(week, column)] = data[(week, column)] / time_decay

        return data


class WeeksLimiter(BaseEstimator, TransformerMixin):
    """ Leave only a certain number of weeks in the dataset.
        Also drop all weeks after the target. """

    def __init__(self, start_week, target_week):
        self.start_week = start_week
        self.target_week = target_week

    def fit(self, data, target=None):
        return self

    def transform(self, data):
        for week in data.columns.get_level_values(0):
            if week < self.start_week or week >= self.target_week:
                data.drop(week, axis=1, inplace=True)

        return data
