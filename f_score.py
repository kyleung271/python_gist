#!/usr/bin/env python3
from functools import partial

import keras.backend as K


def fbeta_score_micro(y_true, y_pred, beta):
    y_true = K.round(K.clip(y_true, 0, 1))
    y_pred = K.round(K.clip(y_pred, 0, 1))

    c1 = K.sum(y_true * y_pred)
    c2 = K.sum(y_pred)
    c3 = K.sum(y_true)

    precision = c1 / (c2 + K.epsilon())
    recall = c1 / (c3 + K.epsilon())

    fbeta_score = (
        (1 + beta ** 2)
        * (precision * recall)
        / (precision * beta ** 2 + recall + K.epsilon())
    )
    return fbeta_score


def fbeta_score_macro(y_true, y_pred, beta):
    y_true = K.round(K.clip(y_true, 0, 1))
    y_pred = K.round(K.clip(y_pred, 0, 1))

    c1 = K.sum(y_true * y_pred, axis=0)
    c2 = K.sum(y_pred, axis=0)
    c3 = K.sum(y_true, axis=0)

    precision = c1 / (c2 + K.epsilon())
    recall = c1 / (c3 + K.epsilon())

    fbeta_score = (
        (1 + beta ** 2)
        * (precision * recall)
        / (precision * beta ** 2 + recall + K.epsilon())
    )

    return K.mean(fbeta_score)


def fbeta_score(beta, weight="micro"):
    """
    Return the f_beta score metric function used by keras.

    >>> y_true = K.constant([[0, 1, 0], [1, 0, 0], [1, 0, 0]])
    >>> y_pred = K.constant([[0, 1, 0], [1, 0, 0], [0, 1, 0]])

    >>> metric = fbeta_score(1, "micro")
    >>> K.eval(metric(y_true, y_pred))
    0.6666666

    >>> metric = fbeta_score(1, "macro")
    >>> K.eval(metric(y_true, y_pred))
    0.4444444

    """
    func = dict(micro=fbeta_score_micro, macro=fbeta_score_macro)[weight]

    func = partial(func, beta=beta)

    func.__name__ = f"f{beta}_score"
    return func


if __name__ == "__main__":
    import doctest

    doctest.testmod()
