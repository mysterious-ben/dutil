from pathlib import Path
import dask
from dask.delayed import Delayed
from typing import Optional, Union

from dutil.src.persist import cached2


def cached_delayed(
    name: Optional[str] = None,
    name_prefix: Optional[str] = None,
    parameters: Optional[dict] = None,
    ignore_args_kwargs: Optional[bool] = None,
    folder: Union[str, Path] = 'cache',
    ftype: str = 'pickle',
    override: bool = False,
    logger=None,
):
    """cached2 + dask.delayed
    
    Returns Delayed object with smart output caching
    """

    def decorator(foo):
        cached_foo = cached2(
            name=name,
            name_prefix=name_prefix,
            parameters=parameters,
            ignore_args_kwargs=ignore_args_kwargs,
            folder=folder,
            ftype=ftype,
            override=override,
            logger=logger,
        )(foo)
        new_foo = dask.delayed()(cached_foo)
        return new_foo
    return decorator


class DelayedParameters():
    """Delayed parameters
    
    Important! Method `set` does not work with dask distributed.
    """

    def __init__(self):
        self._params = {}
        
    def new(self, name: str, value=None) -> Delayed:
        """Return a delayed object for the new parameter"""
        if name in self._params:
            raise KeyError(f'Parameter {name} already exists')
        self._params[name] = value
        return dask.delayed(name=name)(lambda: self._params[name])()
        
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
    datas = tuple(r.get_cache() for r in results)
    return datas
