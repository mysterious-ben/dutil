import pytest
import numpy as np
import pandas as pd
import datetime as dt
import time
from dask import delayed

from dutil.pipeline import cached0, cached, clear_cache, cached_delayed, DelayedParameters, dask_compute


cache_dir = 'cache/temp/'
eps = 0.00001


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
            'd': [pd.Timestamp('2018-01-01'),
                  pd.Timestamp('2018-01-01'),
                  pd.Timestamp('2018-01-01'),
                  pd.Timestamp('2018-01-01'),
                  pd.Timestamp('2018-01-01')],
        }), 'parquet'),
        (pd.DataFrame({
            'a': [0, 1., 3232.22, -1., np.nan],
            'b': [dt.timedelta(hours=1),
                  dt.timedelta(hours=1),
                  dt.timedelta(hours=1),
                  dt.timedelta(hours=1),
                  dt.timedelta(hours=1)],
        }), 'pickle'),
    ]
)
def test_cached_assert_equal(data, ftype):
    @cached0(folder=cache_dir, ftype=ftype, override=False)
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
        (pd.DataFrame({
            'a': [0, 1., 3232.22, -1., np.nan],
            'b': ['a', 'b', 'c', 'ee', '14'],
            'c': [dt.datetime(2018, 1, 1),
                  dt.datetime(2019, 1, 1),
                  dt.datetime(2020, 1, 1),
                  dt.datetime(2021, 1, 1),
                  dt.datetime(2022, 1, 1)],
            'd': [pd.Timestamp('2018-01-01'),
                  pd.Timestamp('2018-01-01'),
                  pd.Timestamp('2018-01-01'),
                  pd.Timestamp('2018-01-01'),
                  pd.Timestamp('2018-01-01')],
        }), 'parquet', 0.1, pd.Timestamp('2018-01-01'),),
        (pd.DataFrame({
            'a': [0, 1., 3232.22, -1., np.nan],
            'b': [dt.timedelta(hours=1),
                  dt.timedelta(hours=1),
                  dt.timedelta(hours=1),
                  dt.timedelta(hours=1),
                  dt.timedelta(hours=1)],
        }), 'pickle', 0.1, pd.Timestamp('2018-01-01'),),
    ]
)
def test_cached_with_args_kwargs_assert_equal(data, ftype, eps, ts):
    @cached0(folder=cache_dir, ftype=ftype, override=False)
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
    'data, output, ftype',
    [
        (pd.DataFrame({
            'a': [0, 1, 3, 5, -1],
            'b': [2, 1, 0, 0, 14],
        }), pd.DataFrame({
            'a': [0, 1, 3, 5, -1],
            'b': [2, 1, 0, 0, 14],
        }), 'parquet'),
        (pd.DataFrame({
            'a': [.5, np.nan, np.nan],
            'b': ['a', 'b', '14'],
            'c': [dt.datetime(2018, 1, 1),
                  dt.datetime(2019, 1, 1),
                  dt.datetime(2022, 1, 1)],
            'd': [pd.Timestamp('2018-01-01'),
                  pd.Timestamp('2018-01-01'),
                  pd.Timestamp('2018-01-01')],
        }), pd.DataFrame({
            'a': [.5],
            'b': ['a'],
            'c': [dt.datetime(2018, 1, 1)],
            'd': [pd.Timestamp('2018-01-01')],
        }), 'parquet'),
        (pd.DataFrame({
            'a': [0, 1., np.nan],
            'b': [dt.timedelta(hours=1),
                  dt.timedelta(hours=1),
                  dt.timedelta(hours=1)],
        }), pd.DataFrame({
            'a': [0, 1.],
            'b': [dt.timedelta(hours=1),
                  dt.timedelta(hours=1)],
        }), 'pickle'),
    ]
)
def test_cached_with_chained_df_assert_equal(data, output, ftype):
    @cached0(folder=cache_dir, ftype=ftype, override=False)
    def load_data():
        return data
    
    @cached0(folder=cache_dir, ftype=ftype, override=False)
    def process_data(df):
        return df.dropna()

    clear_cache(cache_dir)
    df = load_data()
    _ = process_data(df)

    df = load_data()
    processed = process_data(df)

    if isinstance(data, pd.Series):
        pd.testing.assert_series_equal(processed, output)
    elif isinstance(data, pd.DataFrame):
        pd.testing.assert_frame_equal(processed, output)
    elif isinstance(data, np.ndarray):
        np.testing.assert_equal(processed, output)
    else:
        assert processed == output


@pytest.mark.parametrize(
    'data, ftype, eps, ts',
    [
        (pd.DataFrame({
            'a': [0, 1, 3, 5, -1],
            'b': [2, 1, 0, 0, 14],
        }), 'parquet', 0.1, pd.Timestamp('2018-01-01'),),
    ]
)
def test_dask_cached_with_args_kwargs_assert_equal(data, ftype, eps, ts):
    @delayed()
    @cached(folder=cache_dir, ftype=ftype, override=False)
    def load_data(eps, ts):
        assert eps > 0
        assert ts > pd.Timestamp('2000-01-01')
        return data

    clear_cache(cache_dir)
    r = load_data(eps, ts=ts)
    _ = r.compute()
    loaded = r.compute().load()

    if isinstance(data, pd.Series):
        pd.testing.assert_series_equal(loaded, data)
    elif isinstance(data, pd.DataFrame):
        pd.testing.assert_frame_equal(loaded, data)
    elif isinstance(data, np.ndarray):
        np.testing.assert_equal(loaded, data)
    else:
        assert loaded == data


def test_cached_load_time():
    @cached0(folder=cache_dir, override=False)
    def load_data():
        time.sleep(1)
        return 1

    clear_cache(cache_dir)

    start = dt.datetime.utcnow()
    _ = load_data()
    delay = (dt.datetime.utcnow() - start).total_seconds()
    assert delay > 0.95

    start = dt.datetime.utcnow()
    _ = load_data()
    delay = (dt.datetime.utcnow() - start).total_seconds()
    assert delay < 0.95


def test_dask_cached_load_time():
    @delayed()
    @cached(folder=cache_dir, override=False)
    def load_data():
        time.sleep(1)
        return 1

    clear_cache(cache_dir)

    start = dt.datetime.utcnow()
    r = load_data()
    r.compute()
    delay = (dt.datetime.utcnow() - start).total_seconds()
    assert delay > 0.95

    start = dt.datetime.utcnow()
    r = load_data()
    r.compute()
    delay = (dt.datetime.utcnow() - start).total_seconds()
    assert delay < 0.95


def test_dask_pipeline():
    clear_cache(cache_dir)

    @cached_delayed(folder=cache_dir)
    def load_data_1():
        time.sleep(1)
        return 5

    @cached_delayed(folder=cache_dir)
    def load_data_2():
        time.sleep(1)
        return 3

    @cached_delayed(folder=cache_dir)
    def add(x, y):
        return x + y

    d1 = load_data_1()
    d2 = load_data_2()
    r = add(d1, d2)

    start = dt.datetime.utcnow()
    (output,) = dask_compute((r,))
    delay = (dt.datetime.utcnow() - start).total_seconds()
    assert 0.95 < delay < 1.95
    assert output == 8

    start = dt.datetime.utcnow()
    (output,) = dask_compute((r,))
    delay = (dt.datetime.utcnow() - start).total_seconds()
    assert delay < 0.95
    assert output == 8


def test_dask_pipeline_with_parameters():
    clear_cache(cache_dir)

    @cached_delayed(folder=cache_dir)
    def load_data_1(ts: dt.datetime):
        assert ts > dt.datetime(2019, 1, 1)
        time.sleep(1)
        return 5

    @cached_delayed(folder=cache_dir)
    def load_data_2(eps: float):
        time.sleep(1)
        return 3 + eps

    @cached_delayed(folder=cache_dir)
    def add(x, y):
        return x + y

    params = DelayedParameters()
    ts = params.new('ts', value=dt.datetime(2020, 1, 1))
    fix = params.new('fix', value=0.5)
    d1 = load_data_1(ts)
    d2 = load_data_2(fix)
    r = add(d1, d2)

    start = dt.datetime.utcnow()
    (output,) = dask_compute((r,))
    delay = (dt.datetime.utcnow() - start).total_seconds()
    assert 0.95 < delay < 1.95
    assert abs(output - 8.5) < eps

    start = dt.datetime.utcnow()
    (output,) = dask_compute((r,))
    delay = (dt.datetime.utcnow() - start).total_seconds()
    assert delay < 0.95
    assert abs(output - 8.5) < eps

    params.update_many({'ts': dt.datetime(2020, 2, 1), 'fix': 1.5})
    start = dt.datetime.utcnow()
    (output,) = dask_compute((r,))
    delay = (dt.datetime.utcnow() - start).total_seconds()
    assert 0.95 < delay < 1.95
    assert abs(output - 9.5) < eps

    start = dt.datetime.utcnow()
    (output,) = dask_compute((r,))
    delay = (dt.datetime.utcnow() - start).total_seconds()
    assert delay < 0.95
    assert abs(output - 9.5) < eps
