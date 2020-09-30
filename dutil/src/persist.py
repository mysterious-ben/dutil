"""
Data persist tools
"""


import pandas as pd
import dill
from pathlib import Path
import itertools
from typing import *
from loguru import logger as _logger


def _cached_load(ftype, path):
    if ftype == 'parquet':
        data = pd.read_parquet(path)
    elif ftype == 'pickle':
        data = dill.load(open(path, 'rb'))
    else:
        raise ValueError('ftype {} is not recognized'.format(ftype))
    print('data has been loaded from {}'.format(path))
    return data


def _cached_save(data, ftype, path) -> None:
    if ftype == 'parquet':
        data.to_parquet(path, index=False, allow_truncated_timestamps=True)
    elif ftype == 'pickle':
        dill.dump(data, open(path, 'wb'))
    else:
        raise ValueError('ftype {} is not recognized'.format(ftype))


def cached(
    name: Optional[str] = None,
    folder: Union[str, Path] = 'cache',
    ftype: Literal['pickle', 'parquet'] = 'pickle',
    override: bool = False,
    verbose: bool = False,
    logger=None,
):
    """Cache function output on the disk

    :param name: name of the cache file
        if none, name is constructed from the function name and args
    :param folder: name of the cache folder
    :param ftype: type of the cache file
    :param override: if true, override the existing cache file
    :param verbose: if true, log progress
    :param logger: if none, use a new logger
    :return: function output
        loaded from cache file if it exists, generated otherwise
    """

    logger = logger if logger is not None else _logger

    def decorator(load_fun):
        def new_load_fun(*args, **kwargs):
            path = Path(folder)
            path.mkdir(parents=True, exist_ok=True)
            if name is None:
                _n1 = (load_fun.__name__, )
                _n2 = (str(a) for a in args)
                _n3 = (str(k) + str(v) for k, v in kwargs.items())
                _name = '_'.join(itertools.chain(_n1, _n2, _n3)) + '.' + ftype
                path = path / _name
            else:
                path = path / name
            if not override and path.exists():
                data = _cached_load(ftype, path)
                if verbose:
                    logger.info('data has been generated and saved in {}'.format(path))
            else:
                data = load_fun()
                _cached_save(data, ftype, path)
                if verbose:
                    logger.info('data has been loaded from {}'.format(path))
            return data
        return new_load_fun
    return decorator
