"""
Data stats tools
"""

import numpy as np
import pandas as pd
from typing import *


def mean_upper(arr: Union[pd.Series, np.ndarray]) -> float:
    arr = np.asarray(arr)
    return np.mean(np.sort(arr)[len(arr) // 2:])


def mean_lower(arr: Union[pd.Series, np.ndarray]) -> float:
    arr = np.asarray(arr)
    return np.mean(np.sort(arr)[:len(arr) // 2])


def q_01(arr: Union[pd.Series, np.ndarray]) -> float:
    arr = np.asarray(arr)
    return np.quantile(arr, .01)


def q_10(arr: Union[pd.Series, np.ndarray]) -> float:
    arr = np.asarray(arr)
    return np.quantile(arr, .1)


def q_25(arr: Union[pd.Series, np.ndarray]) -> float:
    arr = np.asarray(arr)
    return np.quantile(arr, .25)


def q_50(arr: Union[pd.Series, np.ndarray]) -> float:
    arr = np.asarray(arr)
    return np.quantile(arr, .5)


def q_75(arr: Union[pd.Series, np.ndarray]) -> float:
    arr = np.asarray(arr)
    return np.quantile(arr, .75)


def q_90(arr: Union[pd.Series, np.ndarray]) -> float:
    arr = np.asarray(arr)
    return np.quantile(arr, .90)


def q_99(arr: Union[pd.Series, np.ndarray]) -> float:
    arr = np.asarray(arr)
    return np.quantile(arr, .99)

