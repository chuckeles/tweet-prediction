"""
This module has custom Scikit-learn transformers for making
pipelines and doing grid search.
"""

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

import helpers_models as hm


class ClassBalancer(BaseEstimator, TransformerMixin):
    """ Balances the dataset so both classes have the same amount. """

    def __init__(self):
        self.activeIndices = []
        self.inactiveIndices = []

    def fit(self, data, target=None):
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
    """ Normalizes the dataset so the sums per week are 1. """

    def fit(self, data, target=None):
        return self

    def transform(self, data):
        return hm.normalize_data(data)


class TimeDecayApplier(BaseEstimator, TransformerMixin):
    """ Apply a time decay on the data. Weeks that occurred
        further before the target will have less power. """

    def __init__(self, target_week):
        self.target_week = target_week

    def fit(self, data, target=None):
        return self

    def transform(self, data):
        min_week = data['tweets'].columns.min()
        return hm.apply_time_decay(data, min_week, self.target_week)


class WeeksLimiter(BaseEstimator, TransformerMixin):
    """ Leave only a certain number of weeks in the dataset.
        Also drop all weeks after the target. """

    def __init__(self, first_week, target_week):
        self.first_week = first_week
        self.target_week = target_week

    def fit(self, data, target=None):
        return self

    def transform(self, data):
        min_week = data['tweets'].columns.min()
        max_week = data['tweets'].columns.max()

        # drop weeks before first week
        for week in range(min_week, self.first_week):
            data = data.drop(week, axis=1, level=1)

        # drop weeks after target week
        for week in range(self.target_week, max_week + 1):
            data = data.drop(week, axis=1, level=1)

        return data
