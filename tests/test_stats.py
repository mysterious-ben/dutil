import numpy as np
import pandas as pd
import pytest

from dutil.stats import mean_lower, mean_upper


@pytest.mark.parametrize(
    "data, expected",
    [
        (np.array([0, 1, 5, -1]), 3.0),
        (pd.Series([0, 1, 5, -1]), 3.0),
    ],
)
def test_mean_upper_assert_equal(data, expected):
    actual = mean_upper(data)
    assert actual == expected


@pytest.mark.parametrize(
    "data, expected",
    [
        (np.array([0, 1, 5, -1]), -0.5),
        (pd.Series([0, 1, 5, -1]), -0.5),
    ],
)
def test_mean_lower_assert_equal(data, expected):
    actual = mean_lower(data)
    assert actual == expected
