from __future__ import annotations

import pyarrow
import functools
# from contextlib import contextmanager
# from contextvars import ContextVar
import numpy as np
import pandas as pd
import dill
from pathlib import Path
import shutil
import xxhash
from dask.delayed import Delayed
from loguru import logger as _logger
import json
import multiprocessing
from typing import Optional, Union, List, Tuple, Any

_ = pyarrow.__version__  # explicitly show pyarrow dependency

xxhasher = xxhash.xxh64(seed=42)


def _hash_obj(obj) -> str:
    if isinstance(obj, np.ndarray):
        xxhasher.update(obj.data)
        h = str(xxhasher.intdigest())
        xxhasher.reset()
    elif isinstance(obj, pd.Series):
        try:
            xxhasher.update(obj.values.data)
        except (ValueError, AttributeError):
            xxhasher.update(obj.astype('object').values.data)
        h = str(xxhasher.intdigest())
        xxhasher.reset()
    elif isinstance(obj, pd.DataFrame):
        for c in obj:
            try:
                xxhasher.update(obj[c].values.data)
            except (ValueError, AttributeError):
                xxhasher.update(obj[c].astype('object').values.data)
        h = str(xxhasher.intdigest())
        xxhasher.reset()
    else:
        h = str(obj)
    return h


def _hash_obj_cached(obj) -> str:
    if hasattr(obj, '__cached_hash__'):
        h = obj.__cached_hash__()
    else:
        h = _hash_obj(obj)
    return h


def _kw_is_private(k: str) -> bool:
    return k.startswith('_')


def _get_cache_path(name, name_prefix,
                    parameters, ignore_args, ignore_kwargs,
                    folder, ftype, kwargs_sep, foo, args, kwargs) -> Path:
    path = Path(folder)
    if ignore_args is None:
        ignore_args = parameters is not None
    if ignore_kwargs is None:
        ignore_kwargs = parameters is not None
    if name is None:
        name = foo.__name__
    _n = [name]
    if parameters is not None:
        _n.extend([str(k) + kwargs_sep + _hash_obj_cached(v) for k, v in parameters.items()
                   if not _kw_is_private(k)])
    if not ignore_args:
        _n.extend([_hash_obj_cached(a) for a in args])
    if not ignore_kwargs:
        _n.extend([str(k) + kwargs_sep + _hash_obj_cached(v) for k, v in kwargs.items()
                   if not _kw_is_private(k)])
    elif isinstance(ignore_kwargs, list) or isinstance(ignore_kwargs, set):
        _n.extend([str(k) + kwargs_sep + _hash_obj_cached(v) for k, v in kwargs.items()
                   if k not in ignore_kwargs])
    else:
        assert isinstance(ignore_kwargs, bool)
    full_name = name_prefix + '_'.join(_n) + '.' + ftype
    path = path / full_name
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


def _get_meta_path(cache_path: Path) -> Path:
    return cache_path.parent / (cache_path.name + '.meta')


class CacheMeta:
    """Cache meta data (incl. hash)"""

    def __init__(
        self,
        name: str,
        meta_path: Union[Path],
        cache_path: Union[Path, Tuple[Path]],
        ftype: str,
        nout: Optional[int],
        hash_value: Optional[str],
    ):
        self.name = name
        self.meta_path = meta_path
        self.cache_path = cache_path
        self.ftype = ftype
        self.nout = nout
        self.hash_value = hash_value

    @classmethod
    def from_file(cls, meta_path: Path):
        with open(meta_path, 'rt') as f:
            fields = json.load(f)
        fields['meta_path'] = Path(fields['meta_path'])
        if fields['nout'] is None:
            fields['cache_path'] = Path(fields['cache_path'])
        else:
            fields['cache_path'] = tuple(Path(cp) for cp in fields['cache_path'])
        for k, v in fields.items():
            if isinstance(v, list):
                fields[k] = tuple(v)
        return cls(**fields)

    def dump_to_file(self):
        with open(self.meta_path, 'wt') as f:
            fields = dict(**self.__dict__)
            fields['meta_path'] = str(fields['meta_path'])
            if self.nout is None:
                fields['cache_path'] = str(fields['cache_path'])
            else:
                fields['cache_path'] = [str(cp) for cp in fields['cache_path']]
            json.dump(fields, f)


class CachedResult:
    """Lazy loader for cache data"""

    @classmethod
    def from_user(cls, name, name_prefix, parameters, ignore_args, ignore_kwargs,
                  folder, ftype, kwargs_sep, foo, args, kwargs, logger, nout):
        cp = _get_cache_path(
            name=name, name_prefix=name_prefix, parameters=parameters,
            ignore_args=ignore_args, ignore_kwargs=ignore_kwargs,
            folder=folder, ftype=ftype, kwargs_sep=kwargs_sep,
            foo=foo, args=args, kwargs=kwargs,
        )
        meta_path = _get_meta_path(cp)
        name = cp.name
        if nout is not None:
            cp = tuple(cp.parent / (cp.stem + f'__{i}' + cp.suffix) for i in range(nout))
        if meta_path.exists():
            meta = CacheMeta.from_file(meta_path)
        else:
            meta = CacheMeta(name, meta_path, cp, ftype, nout, hash_value=None)
        return cls(meta=meta, logger=logger)

    def __init__(self, meta, logger):
        self.meta = meta
        self.logger = logger
        self._cache_value = None
        self._lock_dump_load = multiprocessing.Lock()
        self._lock_hash = multiprocessing.Lock()

    def load(self) -> Any:
        """Load data from cache"""

        if self._cache_value is None:
            with self._lock_dump_load:
                if self.meta.nout is None:
                    self._cache_value = _cached_load(self.meta.ftype, self.meta.cache_path)
                else:
                    self._cache_value = tuple(_cached_load(self.meta.ftype, cp) for cp in self.meta.cache_path)
            self.logger.debug('Task {}: data has been loaded from cache'.format(self.meta.name))
        return self._cache_value

    def dump(self, data):
        """Update and dump data to cache"""

        with self._lock_dump_load:
            self._cache_value = data
            if self.meta.nout is None:
                _cached_save(self._cache_value, self.meta.ftype, self.meta.cache_path)
            else:
                assert(len(self._cache_value) == len(self.meta.cache_path) == self.meta.nout)
                for cv, cp in zip(self._cache_value, self.meta.cache_path):
                    _cached_save(cv, self.meta.ftype, cp)
        self.logger.debug('Task {}: data has been saved to cache'.format(self.meta.name))
        self.meta.dump_to_file()

    def __cached_hash__(self):
        """Get hash of cached data

        Used to construct a cache file name
        """

        # Hash may not be required, so it's not automatically computed from data
        if self.meta.hash_value is None:
            with self._lock_hash:
                cache_obj = self.load()  # activates _lock_dump_load
                if self.meta.nout is None:
                    self.meta.hash_value = _hash_obj(cache_obj)
                else:
                    self.meta.hash_value = tuple(_hash_obj(co) for co in cache_obj)
                self.meta.dump_to_file()
            self.logger.debug('Task {}: hash has been computed from data'.format(self.meta.name))
        return self.meta.hash_value

    def exists(self):
        return self.meta.meta_path.exists()


class CachedResultItem:
    def __init__(self, result: CachedResult, item: Any):
        self.result = result
        self.item = item
    
    def load(self) -> Any:
        if self.item is None:
            return self.result.load()
        else:
            return self.result.load()[self.item]

    def dump(self, data):
        self.result.dump(data)

    def __cached_hash__(self):
        if self.item is None:
            return self.result.__cached_hash__()
        else:
            return self.result.__cached_hash__()[self.item]


def cached(
    name: Optional[str] = None,
    name_prefix: str = '',
    parameters: Optional[dict] = None,
    ignore_args: Optional[bool] = None,
    ignore_kwargs: Optional[Union[bool, List[str]]] = None,
    folder: Union[str, Path] = 'cache',
    ftype: str = 'pickle',
    kwargs_sep: str = '|',
    nout: Optional[int] = None,
    override: bool = False,
    logger=None,
):
    """Cache function output on the disk

    Features:
    - Pickle and parquet serialization
    - Special treatment for Delayed objects
    - Lazy cache loading
    - Hashing of complex arguments

    :param name: name of the cache file
        if none, name is constructed from the function name and args
    :param parameters: include these parameters in the name
        only meaningful when `name=None`
        Improtant: parameters starting with _ (underscore) will be ignored
    :param ignore_args: if true, do not add args to the name
    :param ignore_kwargs: if true, do not add kwargs to the name
        it's also possible to specify a list of kwargs to ignore
        Improtant: kwargs starting with _ (underscore) will be ignored
    :param folder: name of the cache folder
    :param ftype: type of the cache file
        'pickle' | 'parquet'
    :param kwargs_sep: string separating a keyword parameter and its value
    :param override: if true, override the existing cache file
    :param logger: if none, use a new logger
    :return: new function
        output is lazily loaded from cache file if it exists, generated otherwise
        .load() to get data
    """

    logger = logger if logger is not None else _logger

    def decorator(foo):
        """Cache function output on the disk"""
        @functools.wraps(foo)
        def new_foo(*args, **kwargs):
            result = CachedResult.from_user(
                name, name_prefix, parameters, ignore_args, ignore_kwargs,
                folder, ftype, kwargs_sep, foo, args, kwargs, logger, nout
            )
            if not override and result.exists():
                # if the result (= cache OR cache + hash) exists, do nothing - just pass it on
                # the cache will be loaded only if required later
                if nout is not None:
                    output = tuple(CachedResultItem(result, i) for i in range(nout))
                else:
                    output = CachedResultItem(result, None)
                logger.info('Task {}: skip (cache exists)'.format(result.meta.name))
            else:
                # if the result does not exist, generate data and save cache
                dask_args_detected = any(isinstance(a, Delayed) for a in args)
                dask_kwargs_detected = any(isinstance(v, Delayed) for _, v in kwargs.items())
                if not dask_args_detected and not dask_kwargs_detected:
                    # eager load cache for all arguments
                    args = [a.load() if isinstance(a, CachedResultItem) else a for a in args]
                    kwargs = {k: v.load() if isinstance(v, CachedResultItem) else v for k, v in kwargs.items()}
                    data = foo(*args, **kwargs)
                    result.dump(data)
                    if nout is not None:
                        output = tuple(CachedResultItem(result, i) for i in range(nout))
                    else:
                        output = CachedResultItem(result, None)
                    logger.info('Task {}: data has been computed and saved to cache'.format(result.meta.name))
                else:
                    # if any of the arguments is a Delayed object, return anything
                    output = foo(*args, **kwargs)
            return output
        return new_foo
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
