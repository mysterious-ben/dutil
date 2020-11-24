import dask
from dask.delayed import Delayed
from contextlib import contextmanager
import multiprocessing
from typing import Any, Optional

from dutil.pipeline import CachedResultItem
from dutil.pipeline._cached import _kw_is_private


class DelayedParameter:
    """Delayed parameter = a Delayed object that can change the returned value
    
    Important! Methods `update` and `context` do not work with dask distributed.

    :param name: parameter name
    :param value: parameter value
    """
    def __init__(self, name, value=None):
        self._name = name
        self._value = value
        self._delayed = dask.delayed(name=name)(lambda: self._value)()
        self._lock = multiprocessing.Lock()
        
    def set(self, value) -> None:
        """Permanently change the value of this parameter"""
        self._value = value
        
    def __call__(self) -> Delayed:
        """Get a Delayed object"""
        return self._delayed

    @contextmanager
    def context(self, value):
        """Change the value of this parameter within a context"""
        with self._lock:
            old_value = self._value
            self.set(value)
            yield
            self.set(old_value)


class DelayedParameters():
    """A dictionary of delayed parameters
    
    Important! Methods `update` and `context` do not work with dask distributed.
    """

    def __init__(self, params: Optional[dict] = None):
        self._params = {}
        self._param_delayed = {}
        if params is not None:
            for name, value in params.items():
                self._params[name] = value
                self._param_delayed[name] = dask.delayed(name=name)(lambda: self._params[name])()
        self._lock = multiprocessing.Lock()
    
    def create(self, name: str, value: Any = None) -> Delayed:
        """Create a new parameter and return a delayed object"""
        if name in self._params:
            raise KeyError(f'Parameter {name} already exists')
        self._params[name] = value
        self._param_delayed[name] = dask.delayed(name=name)(lambda: self._params[name])()
        return self._param_delayed[name]

    def create_many(self, d: dict) -> None:
        """Create multiple parameters at once"""
        for k, v in d.items():
            self.create(k, v)

    def get_params(self) -> dict:
        """Get parameters as a dictionary (name -> value)"""
        return {k: v for k, v in self._params.items() if not _kw_is_private(k)}

    def get_delayed(self, name: str) -> Delayed:
        """Get a Delayed object for the chosen parameter"""
        return self._param_delayed[name]

    def update(self, name: str, value: Any) -> None:
        """Permanently update parameter value"""
        if name not in self._params:
            raise KeyError(f'Parameter {name} does not exist')
        self._params[name] = value

    def update_many(self, d: dict) -> None:
        """Permanently update multiple parameter values"""
        for k, v in d.items():
            self.update(k, v)

    @contextmanager
    def context(self, d: dict):
        """Update multiple parameter values within a context"""
        with self._lock:
            old_params = dict(**self._params)
            self.update_many(d)
            yield
            self.update_many(old_params)


def delayed_compute(tasks, scheduler='threads') -> tuple:
    """Compute values of Delayed objects or load it from cache"""
    results = dask.compute(*tasks, scheduler=scheduler)
    datas = tuple(r.load() if isinstance(r, CachedResultItem) else r for r in results)
    return datas
