"""
Data transform utilities
"""

import numpy as np
import pandas as pd


def ht(arr, n: int = 2):
    """Get first and last (top and bottom) entries

    :param arr: DataFrame, Series, array, list, or tuple
    :param n: number of top and bottom entries
    :return: concatenated top and bottom entries
    """

    if len(arr) <= n*2:
        return arr
    elif isinstance(arr, pd.DataFrame) or isinstance(arr, pd.Series):
        return pd.concat([arr.head(n), arr.tail(n)])
    elif isinstance(arr, np.ndarray):
        return np.concatenate([arr[:n], arr[-n:]])
    elif isinstance(arr, list) or isinstance(arr, tuple):
        return arr[:n] + arr[-n:]
    else:
        raise ValueError(f'{arr} type is not recognized')
