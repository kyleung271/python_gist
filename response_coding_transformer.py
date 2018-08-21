#!/usr/bin/env python3
from functools import partial

import numpy as np
import pandas as pd
import scipy.sparse as sprs
from sklearn.base import BaseEstimator, TransformerMixin

epsilon = np.finfo(float).eps


def sorted_unique(x):
    return pd.Series(x).value_counts().index.values


def naive_count(count):
    """
    Calculate probability distribution from occurrence vector.

    >>> naive_count([1, 1])
    array([0.5, 0.5])
    """
    count = np.atleast_1d(count)
    return count / (count.sum() + epsilon)


def additive_smoothing(count, alpha):
    """
    Calculate probability distribution from occurrence vector with additive smoothing.

    >>> additive_smoothing([2, 1], 1)
    array([0.6, 0.4])
    """
    count = np.atleast_1d(count)
    return (count + alpha) / (count.sum() + count.size * alpha)


class ResponseCodingTransformer(BaseEstimator, TransformerMixin):
    """
    Simple Response coding.

    >>> from sklearn.pipeline import make_pipeline
    >>> from sklearn.preprocessing import OneHotEncoder
    >>> from sklearn.feature_extraction.text import CountVectorizer

    >>> rst = ResponseCodingTransformer()

    Categorical data can be applied through OneHotEncoder.

    >>> encoder = make_pipeline(OneHotEncoder(), rst)
    >>> encoder.fit_transform([[1], [2], [1]], ["a", "b", "b"])
    matrix([[0.5, 0.5],
            [1. , 0. ],
            [0.5, 0.5]])

    The corresponding y value can be accessed via:
    >>> rst.token
    array(['b', 'a'], dtype=object)

    Custom probability distribution estimator can be used.
    Any extra keyword argumants are passed through.

    >>> rst = ResponseCodingTransformer(additive_smoothing, alpha=1)
    >>> encoder = make_pipeline(OneHotEncoder(), rst)
    >>> encoder.fit_transform([[1], [2], [1]], ["a", "b", "b"])
    matrix([[0.5       , 0.5       ],
            [0.66666667, 0.33333333],
            [0.5       , 0.5       ]])

    Text data can be applied through CountVectorizer.

    >>> encoder = make_pipeline(CountVectorizer(), rst)
    >>> encoder.fit_transform(["Testing", "Testing again"], [1, 0])
    matrix([[0.5       , 0.5       ],
            [0.40824829, 0.57735027]])

    >>> encoder.transform(["foo", "foo again"])
    matrix([[0.        , 0.        ],
            [0.33333333, 0.66666667]])
    """

    def __init__(self, prob_func=naive_count, **kwargs):
        # prob_func warpped in list to prevent BaseEstimator messing with the closure
        self.prob_func = [partial(prob_func, **kwargs)]

    def fit(self, X, y):
        y = np.atleast_1d(y)
        self.token = sorted_unique(y)

        m, n = X.shape
        o = self.token.size

        token_to_index = {t: i for i, t in enumerate(self.token)}

        count = np.zeros((n, o), dtype=int)

        for (i, j, c) in zip(*sprs.find(X)):
            k = token_to_index[y[i]]
            count[j, k] += c

        prob = np.apply_along_axis(self.prob_func[0], 1, count)

        self.log2_p = np.log2(prob)
        return self

    def transform(self, X):
        return np.exp2((X @ self.log2_p - epsilon) / X.sum(axis=1).reshape(-1, 1))


if __name__ == "__main__":
    import doctest

    doctest.testmod()
