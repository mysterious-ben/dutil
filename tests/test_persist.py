import pytest
import numpy as np
import pandas as pd
import datetime as dt
from dask import delayed

from dutil.src.persist import cached, dask_cached, clear_cache

cache_dir = 'cache/temp/'


@pytest.mark.parametrize(
    'data, ftype',
    [
        ((0, 1, 3, 5, -1), 'pickle'),
        ((0, 1., 3232.22, 5., -1., None), 'pickle'),
        ([0, 1, 3, 5, -1], 'pickle'),
        ([0, 1., 3232.22, 5., -1., None], 'pickle'),
        (pd.Series([0, 1, 3, 5, -1]), 'pickle'),
        (pd.Series([0, 1., 3232.22, 5., -1., np.nan]), 'pickle'),
        (pd.DataFrame({
            'a': [0, 1, 3, 5, -1],
            'b': [2, 1, 0, 0, 14],
        }), 'pickle'),
        (pd.DataFrame({
            'a': [0, 1., 3232.22, -1., np.nan],
            'b': ['a', 'b', 'c', 'ee', '14'],
            'c': [dt.datetime(2018, 1, 1),
                  dt.datetime(2019, 1, 1),
                  dt.datetime(2020, 1, 1),
                  dt.datetime(2021, 1, 1),
                  dt.datetime(2022, 1, 1)],
        }), 'pickle'),
        (pd.DataFrame({
            'a': [0, 1, 3, 5, -1],
            'b': [2, 1, 0, 0, 14],
        }), 'parquet'),
        (pd.DataFrame({
            'a': [0, 1., 3232.22, -1., np.nan],
            'b': ['a', 'b', 'c', 'ee', '14'],
            'c': [dt.datetime(2018, 1, 1),
                  dt.datetime(2019, 1, 1),
                  dt.datetime(2020, 1, 1),
                  dt.datetime(2021, 1, 1),
                  dt.datetime(2022, 1, 1)],
        }), 'parquet'),
    ]
)
def test_cached_assert_equal(data, ftype):
    @cached(folder=cache_dir, ftype=ftype, override=False, verbose=True)
    def load_data():
        return data

    clear_cache(cache_dir)
    _ = load_data()
    loaded = load_data()

    if isinstance(data, pd.Series):
        pd.testing.assert_series_equal(loaded, data)
    elif isinstance(data, pd.DataFrame):
        pd.testing.assert_frame_equal(loaded, data)
    elif isinstance(data, np.ndarray):
        np.testing.assert_equal(loaded, data)
    else:
        assert loaded == data


@pytest.mark.parametrize(
    'data, ftype, eps, ts',
    [
        (pd.DataFrame({
            'a': [0, 1, 3, 5, -1],
            'b': [2, 1, 0, 0, 14],
        }), 'parquet', 0.1, pd.Timestamp('2018-01-01'),),
    ]
)
def test_cached_with_arges_kwargs_assert_equal(data, ftype, eps, ts):
    @cached(folder=cache_dir, ftype=ftype, override=False, verbose=True)
    def load_data(eps, ts):
        assert eps > 0
        assert ts > pd.Timestamp('2000-01-01')
        return data

    clear_cache(cache_dir)
    _ = load_data(eps, ts=ts)
    loaded = load_data(eps, ts=ts)

    if isinstance(data, pd.Series):
        pd.testing.assert_series_equal(loaded, data)
    elif isinstance(data, pd.DataFrame):
        pd.testing.assert_frame_equal(loaded, data)
    elif isinstance(data, np.ndarray):
        np.testing.assert_equal(loaded, data)
    else:
        assert loaded == data


@pytest.mark.parametrize(
    'data, ftype, eps, ts',
    [
        (pd.DataFrame({
            'a': [0, 1, 3, 5, -1],
            'b': [2, 1, 0, 0, 14],
        }), 'parquet', 0.1, pd.Timestamp('2018-01-01'),),
    ]
)
def test_dask_cached_with_arges_kwargs_assert_equal(data, ftype, eps, ts):
    @delayed(pure=False)
    @dask_cached(folder=cache_dir, ftype=ftype, override=False, verbose=True)
    def load_data(eps, ts):
        assert eps > 0
        assert ts > pd.Timestamp('2000-01-01')
        return data

    clear_cache(cache_dir)
    r = load_data(eps, ts=ts)
    _ = r.compute()
    loaded = r.compute()

    if isinstance(data, pd.Series):
        pd.testing.assert_series_equal(loaded, data)
    elif isinstance(data, pd.DataFrame):
        pd.testing.assert_frame_equal(loaded, data)
    elif isinstance(data, np.ndarray):
        np.testing.assert_equal(loaded, data)
    else:
        assert loaded == data
