"""
Data persist tools
"""


import pyarrow
import numpy as np
import pandas as pd
import dill
from pathlib import Path
import shutil
import xxhash
from dask.delayed import Delayed
from typing import Optional, Union
from loguru import logger as _logger

_ = pyarrow.__version__  # explicitly show pyarrow dependency

xxhasher = xxhash.xxh64(seed=42)


def _hash_obj(obj):
    if isinstance(obj, np.ndarray):
        xxhasher.update(obj.data)
        h = str(xxhasher.intdigest())
        xxhasher.reset()
    elif isinstance(obj, pd.Series):
        xxhasher.update(obj.values.data)
        h = str(xxhasher.intdigest())
        xxhasher.reset()
    elif isinstance(obj, pd.DataFrame):
        for c in obj:
            xxhasher.update(obj[c].values.data)
        h = str(xxhasher.intdigest())
        xxhasher.reset()
    else:
        h = str(obj)
    return h


def _get_cache_path(name, parameters, ignore_args_kwargs,
                    folder, ftype, foo, args, kwargs):
    path = Path(folder)
    if ignore_args_kwargs is None:
        ignore_args_kwargs = (parameters is not None)
    if name is None:
        _n = [foo.__name__, ]
        if parameters is not None:
            _n.extend([str(k) + _hash_obj(v) for k, v in parameters.items()])
        if not ignore_args_kwargs:
            _n.extend([_hash_obj(a) for a in args])
            _n.extend([str(k) + _hash_obj(v) for k, v in kwargs.items()])
        _name = '_'.join(_n) + '.' + ftype
        path = path / _name
    else:
        path = path / name
    return path


def _cached_load(ftype, path):
    if ftype == 'parquet':
        data = pd.read_parquet(path)
    elif ftype == 'pickle':
        data = dill.load(open(path, 'rb'))
    else:
        raise ValueError('ftype {} is not recognized'.format(ftype))
    return data


def _cached_save(data, ftype, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if ftype == 'parquet':
        data.to_parquet(path, index=False, allow_truncated_timestamps=True)
    elif ftype == 'pickle':
        dill.dump(data, open(path, 'wb'))
    else:
        raise ValueError('ftype {} is not recognized'.format(ftype))


def cached(
    name: Optional[str] = None,
    parameters: Optional[dict] = None,
    ignore_args_kwargs: Optional[bool] = None,
    folder: Union[str, Path] = 'cache',
    ftype: str = 'pickle',
    override: bool = False,
    verbose: bool = False,
    logger=None,
):
    """Cache function output on the disk

    :param name: name of the cache file
        if none, name is constructed from the function name and args
    :param parameters: include these parameters in the name
        only meaningful when `name=None`
    :param ignore_args_kwargs: if true, do not add args and kwargs to the name
        only meaningful when `name=None`
    :param folder: name of the cache folder
    :param ftype: type of the cache file
        'pickle' | 'parquet'
    :param override: if true, override the existing cache file
    :param verbose: if true, log progress
    :param logger: if none, use a new logger
    :return: new function
        output is loaded from cache file if it exists, generated otherwise
    """

    logger = logger if logger is not None else _logger

    def decorator(foo):
        def new_load_fun(*args, **kwargs):
            path = _get_cache_path(name, parameters, ignore_args_kwargs,
                                   folder, ftype, foo, args, kwargs)
            if not override and path.exists():
                data = _cached_load(ftype, path)
                if verbose:
                    logger.info('data has been generated and saved in {}'.format(path))
            else:
                data = foo(*args, **kwargs)
                _cached_save(data, ftype, path)
                if verbose:
                    logger.info('data has been loaded from {}'.format(path))
            return data
        return new_load_fun
    return decorator


def dask_cached(
    name: Optional[str] = None,
    parameters: Optional[dict] = None,
    ignore_args_kwargs: Optional[bool] = None,
    folder: Union[str, Path] = 'cache',
    ftype: str = 'pickle',
    override: bool = False,
    verbose: bool = False,
    logger=None,
):
    """Cache function output on the disk (works with dask.delayed)

    :param name: name of the cache file
        if none, name is constructed from the function name and args
    :param parameters: include these parameters in the name
        only meaningful when `name=None`
    :param ignore_args_kwargs: if true, do not add args and kwargs to the name
        only meaningful when `name=None`
    :param folder: name of the cache folder
    :param ftype: type of the cache file
        'pickle' | 'parquet'
    :param override: if true, override the existing cache file
    :param verbose: if true, log progress
    :param logger: if none, use a new logger
    :return: new function
        output is loaded from cache file if it exists, generated otherwise
    """

    logger = logger if logger is not None else _logger

    def decorator(foo):
        def new_load_fun(*args, **kwargs):
            path = _get_cache_path(name, parameters, ignore_args_kwargs,
                                   folder, ftype, foo, args, kwargs)
            if not override and path.exists():
                data = _cached_load(ftype, path)
                if verbose:
                    logger.info('data has been generated and saved in {}'.format(path))
            else:
                data = foo(*args, **kwargs)
                dask_args = any(isinstance(a, Delayed) for a in args)
                dask_kwargs = any(isinstance(v, Delayed) for k, v in kwargs.items())
                if not dask_args and not dask_kwargs:
                    _cached_save(data, ftype, path)
                    if verbose:
                        logger.info('data has been loaded from {}'.format(path))
            return data
        return new_load_fun
    return decorator


def clear_cache(
    folder: Union[str, Path] = 'cache',
    ignore_errors: bool = True,
):
    """Clear the cache folder

     :param folder: name of the cache folder
    """
    folder = Path(folder).absolute()
    shutil.rmtree(folder, ignore_errors=ignore_errors)
