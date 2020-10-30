import dask
from dask.delayed import Delayed

from dutil.pipeline import CachedResult


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
