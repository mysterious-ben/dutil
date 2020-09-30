import pytest
import numpy as np
import pandas as pd
import datetime as dt

from dutil.src.stats import mean_upper, mean_lower


@pytest.mark.parametrize(
    'data, expected',
    [
        (np.array([0, 1, 5, -1]), 3.),
        (pd.Series([0, 1, 5, -1]), 3.),
    ]
)
def test_mean_upper_assert_equal(data, expected):
    actual = mean_upper(data)
    assert actual == expected


@pytest.mark.parametrize(
    'data, expected',
    [
        (np.array([0, 1, 5, -1]), -.5),
        (pd.Series([0, 1, 5, -1]), -.5),
    ]
)
def test_mean_lower_assert_equal(data, expected):
    actual = mean_lower(data)
    assert actual == expected
