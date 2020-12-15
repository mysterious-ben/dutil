# dutil

A few data utilities to make life of a data scientist easier

## Installation

```shell
pip install dutil
```

## Modules

- `pipeline` (data caching and pipelines)
- `stats` (statistical functions)
- `string` (string manipulations)
- `transform` (data transformations)
- `jupyter` (tools for jupyter notebooks)


### Pipeline

```python
import dutil.pipeline as dpipe
import pandas as pd
import numpy as np
from loguru import logger

# --- Define data transformations via step functions (similar to dask.delayed)

@dpipe.delayed_cached()  # lazy computation + caching on disk
def load_1():
    df = pd.DataFrame({'a': [1., 2.], 'b': [0.1, np.nan]})
    logger.info('Loaded {} records'.format(len(df)))
    return df

@dpipe.delayed_cached()  # lazy computation + caching on disk
def load_2(timestamp):
    df = pd.DataFrame({'a': [0.9, 3.], 'b': [0.001, 1.]})
    logger.info('Loaded {} records'.format(len(df)))
    return df

@dpipe.delayed_cached()  # lazy computation + caching on disk
def compute(x, y, eps):
    assert x.shape == y.shape
    diff = ((x - y).abs() / (y.abs()+eps)).mean().mean()
    logger.info('Difference is computed')
    return diff

# Define pipeline dependencies
ts = pd.Timestamp(2019, 1, 1)
eps = 0.01
s1 = load_1()
s2 = load_2(ts)
diff = compute(s1, s2, eps)

# Trigger pipeline execution
print('diff: {:.3f}'.format(dpipe.delayed_compute((diff, ))[0]))
```

### Stats

```python
from dutil.stats import mean_lower, mean_upper
import pandas as pd
ss = pd.Series([0, 1, 5, -1])
mean_lower(ss)  # Compute mean among 50% smallest elements
mean_upper(ss)  # Compute mean among 50% biggest elements
```

### String

```python
from dutil.string import compare_companies
compare_companies("Aarons Holdings Company Inc.", "Aaron's, Inc.")  # Give match rating for two company names
```

### Transform

```python
from dutil.transform import ht
import pandas as pd
df = pd.DataFrame({'a': [0, 2, 2, 4, 6], 'b': [1, 1, 1, 1, 1]})
ht(df)  # Return first and last rows of a DataFrame, a Series, or an array
```

### Jupyter

```python
from dutil.jupyter import dht
import pandas as pd
df = pd.DataFrame({'a': [0, 2, 2, 4, 6], 'b': [1, 1, 1, 1, 1]})
dht(df)  # Display first and last rows of a DataFrame, a Series, or an array in a Jupyter notebook
```
