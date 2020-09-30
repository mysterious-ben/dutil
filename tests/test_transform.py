import pytest
import numpy as np
import pandas as pd
import datetime as dt

from dutil.src.transform import ht


@pytest.mark.parametrize(
    'data, expected',
    [
        ((0, 1, 3, 5, -1), (0, 1, 5, -1)),
        ([0, 1, 3, 5, -1], [0, 1, 5, -1]),
        ([0, 1., 3232.22, 5., -1., np.nan], [0, 1., -1., np.nan]),
        (np.array([0, 1, 3, 5, -1]), np.array([0, 1, 5, -1])),
        (np.array([0, 1., 3232.22, 5., -1., np.nan]), np.array([0, 1., -1., np.nan])),
        (pd.Series([0, 1, 3, 5, -1]),
         pd.Series([0, 1, 5, -1], index=[0, 1, 3, 4])),
        (pd.Series([0, 1., 3232.22, -1., np.nan]),
         pd.Series([0, 1., -1., np.nan], index=[0, 1, 3, 4])),
        (pd.DataFrame({
            'a': [0, 1, 3, 5, -1],
            'b': [2, 1, 0, 0, 14],
        }),
         pd.DataFrame({
             'a': [0, 1, 5, -1],
             'b': [2, 1, 0, 14],
         }, index=[0, 1, 3, 4])),
        (pd.DataFrame({
            'a': [0, 1., 3232.22, -1., np.nan],
            'b': ['a', 'b', 'c', 'ee', '14'],
            'c': [dt.datetime(2018, 1, 1),
                  dt.datetime(2019, 1, 1),
                  dt.datetime(2020, 1, 1),
                  dt.datetime(2021, 1, 1),
                  dt.datetime(2022, 1, 1)],
        }),
         pd.DataFrame({
             'a': [0, 1., -1., np.nan],
             'b': ['a', 'b', 'ee', '14'],
             'c': [dt.datetime(2018, 1, 1),
                   dt.datetime(2019, 1, 1),
                   dt.datetime(2021, 1, 1),
                   dt.datetime(2022, 1, 1)],
         }, index=[0, 1, 3, 4])
        ),
        ((0,), (0,)),
        (np.array([0]), np.array([0])),
        (pd.Series([0]), pd.Series([0])),
        (pd.DataFrame({
            'a': [0],
            'b': [2],
        }),
        pd.DataFrame({
            'a': [0],
            'b': [2],
        })),
    ]
)
def test_ht_assert_equal(data, expected):
    actual = ht(data, n=2)
    if isinstance(data, pd.Series):
        pd.testing.assert_series_equal(actual, expected)
    elif isinstance(data, pd.DataFrame):
        pd.testing.assert_frame_equal(actual, expected)
    elif isinstance(data, np.ndarray):
        np.testing.assert_equal(actual, expected)
    else:
        assert actual == expected
