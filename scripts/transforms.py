"""
This module has custom Scikit-learn transformers for making
pipelines and doing grid search.
"""

from sklearn.base import BaseEstimator, TransformerMixin

from helpers_models import *


class TargetMaker(BaseEstimator, TransformerMixin):
    """ Adds a target column to the data. Can't be use for predictors. """

    def __init__(self, target_week):
        self.target_week = target_week

    def fit(self, data, target=None):
        return self

    def transform(self, data):
        return make_target(data, self.target_week)


class ClassBalancer(BaseEstimator, TransformerMixin):
    """ Balances the dataset so both classes have the same amount. Can't be used for predictors. """

    def fit(self, data, target=None):
        return self

    def transform(self, data):
        return balance_data(data)


class Normalizer(BaseEstimator, TransformerMixin):
    """ Normalizes the dataset so the sums per week are 1. """

    def __init__(self):
        self.target = pd.DataFrame()

    def fit(self, data, target=None):
        # FIXME: This does not work as expected
        # if 'target' in data:
        #     # save the target so we can restore it later
        #     self.target = pd.DataFrame(data['target'])
        #
        #     # fix the column indices
        #     self.target.columns = pd.MultiIndex.from_tuples([(c, 0) for c in self.target])

        return self

    def transform(self, data):
        # FIXME: This does not work as expected
        # if 'target' in data:
        #     # balance only input data
        #     balanced = balance_data(data.drop('target', axis=1, level=0))
        #
        #     # concatenate with target
        #     return pd.concat([balanced, self.target], axis=1)
        #
        # else:
        return balance_data(data)


class TimeDecayApplier(BaseEstimator, TransformerMixin):
    """ Apply a time decay on the data. Weeks more into the past will have lower numbers. """

    def __init__(self, first_week, target_week):
        self.first_week = first_week
        self.target_week = target_week

    def fit(self, data, target=None):
        return self

    def transform(self, data):
        return apply_time_decay(data, self.first_week, self.target_week)


class WeeksLimiter(BaseEstimator, TransformerMixin):
    """ Leave only a certain number of weeks in the dataset.
        Also drop all weeks after the target. """

    def __init__(self, first_week, target_week):
        self.min_week = 0
        self.max_week = 0
        self.first_week = first_week
        self.target_week = target_week

    def fit(self, data, target=None):
        self.min_week = data['tweets'].columns.min()
        self.max_week = data['tweets'].columns.max()

        return self

    def transform(self, data):
        # drop weeks before first week
        for week in range(self.min_week, self.first_week):
            data = data.drop(week, axis=1, level=1)

        # drop weeks after target week
        for week in range(self.target_week, self.max_week + 1):
            data = data.drop(week, axis=1, level=1)
