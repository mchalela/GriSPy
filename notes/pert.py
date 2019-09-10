#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division

import numpy as np


def clean(data):
    """Convert data as numpy array if necesary

    If data is instance of numpy.ndarray the same object is returned
    If data has attribute '__iter__' is converted to numpy.ndarray
    A least if no condition is true the same object is returned

    """
    if isinstance(data, np.ndarray):
        return data
    elif hasattr(data, "__iter__"):
        return np.array(data)
    return np.array([data])


def expect(o, p, m):
    """expected time: the best estimate of the time required to accomplish a
    task, accounting for the fact that things don't always proceed as normal
    (the implication being that the expected time is the average time the
    task would require if the task were repeated on a number of occasions over
    an extended period of time).

    ::

            TE = (O + 4M + P) ÷ 6

    Where:

    - optimistic time (O): the minimum possible time required to accomplish a
      task, assuming everything proceeds better than is normally expected
    - pessimistic time (P): the maximum possible time required to accomplish a
      task, assuming everything goes wrong (but excluding major catastrophes).
    - most likely time (M): the best estimate of the time required to
      accomplish a task, assuming everything proceeds as normal.

    """
    o, m , p = clean(o), clean(m), clean(p)
    return (o + 4 * m + p) / 6


def std(o, p):
    """Standar deviation.

    ::

        S = (P - O) / 6

    Where:

    - optimistic time (O): the minimum possible time required to accomplish a
      task, assuming everything proceeds better than is normally expected
    - pessimistic time (P): the maximum possible time required to accomplish a
      task, assuming everything goes wrong (but excluding major catastrophes).

    """
    o, p = clean(o),  clean(p)
    return (p - o) / 6


def var(o, p):
    """Variation

    ::

        S = ((P - O) / 6) ^ 2

    Where:

    - optimistic time (O): the minimum possible time required to accomplish a
      task, assuming everything proceeds better than is normally expected
    - pessimistic time (P): the maximum possible time required to accomplish a
      task, assuming everything goes wrong (but excluding major catastrophes).

    """
    return std(o, p) ** 2



def estimate(o, p, m):
    """Create a estimation for a set of values

    # 68.2%, 95.4%, 99.7%
    r68, r95, r99 = estimate([1, 2], [3, 4], [5, 6])


    """
    es, stds = np.sum(expect(o, p, m)), np.sum(std(o, p))
    return np.array([
        [es-stds if es-stds > 0 else 0, es+stds*2],
        [es-stds*2 if es-stds*2 > 0 else 0, es+stds*2],
        [es-stds*3 if es-stds*3 > 0 else 0, es+stds*3],
    ])
