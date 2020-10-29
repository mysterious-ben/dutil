from pathlib import Path
import dask
from dask.delayed import Delayed
from typing import Optional, Union, List

from dutil.pipeline import cached, CachedResult


def cached_delayed(
    name: Optional[str] = None,
    name_prefix: Optional[str] = None,
    parameters: Optional[dict] = None,
    ignore_args: Optional[bool] = None,
    ignore_kwargs: Optional[Union[bool, List[str]]] = None,
    folder: Union[str, Path] = 'cache',
    ftype: str = 'pickle',
    override: bool = False,
    logger=None,
):
    """cached + dask.delayed
    
    Returns Delayed object with smart output caching
    """

    def decorator(foo):
        cached_foo = cached(
            name=name,
            name_prefix=name_prefix,
            parameters=parameters,
            ignore_args=ignore_args,
            ignore_kwargs=ignore_kwargs,
            folder=folder,
            ftype=ftype,
            override=override,
            logger=logger,
        )(foo)
        new_foo = dask.delayed()(cached_foo)
        return new_foo
    return decorator


class DelayedParameter:
    def __init__(self, name, value=None):
        self._name = name
        self._value = value
        self._delayed = dask.delayed(lambda: self._value)()
        
    def set(self, value):
        self._value = value
        
    def __call__(self):
        return self._delayed


class DelayedParameters():
    """Delayed parameters
    
    Important! Method `set` does not work with dask distributed.
    """

    def __init__(self):
        self._params = {}
        self._param_delayed = {}
    
    def get_params(self):
        return self._params

    def new(self, name: str, value=None) -> Delayed:
        """Return a delayed object for the new parameter"""
        if name in self._params:
            raise KeyError(f'Parameter {name} already exists')
        self._params[name] = value
        self._param_delayed[name] = dask.delayed(name=name)(lambda: self._params[name])()
        return self._param_delayed[name]

    def update(self, name: str, value) -> None:
        """Update parameter value"""
        if name not in self._params:
            raise KeyError(f'Parameter {name} does not exist')
        self._params[name] = value

    def update_many(self, d: dict) -> None:
        """Update multiple parameters values"""
        for k, v in d.items():
            self.update(k, v)


def dask_compute(tasks, scheduler='threads') -> tuple:
    """Compute values of Delayed objects"""
    results = dask.compute(*tasks, scheduler=scheduler)
    datas = tuple(r.load() if isinstance(r, CachedResult) else r for r in results)
    return datas
