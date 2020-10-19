"""
Dask pipeline tools
"""

import time
import datetime as dt

from dutil.src.dask_pipeline import cached_delayed, DelayedParameters, dask_compute
from dutil.src.persist import clear_cache

cache_dir = 'cache/temp/'
eps = 0.0001


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
