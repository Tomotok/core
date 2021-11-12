## Commit message rules
 - start with verb in infinitive form
 - be reasonably specific
 - e.g. "Add new sinogram plotting function to tools/sinogram.py"
 - add `WIP` to the begining of commit message if changes are not ready for implementation

## Importing convetions
- never use star imports eg.
```python
# NEVER use
from numpy import *
# correct way
import numpy as np
```

### shortcuts for modules
- np for numpy
- plt for matplotlib.pyplot

## Docstrings

 - Use numpy style docstrings, example below   
https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html

 - Skip first line in multiline docstrings.
 - Write documentation for \_\_init\_\_ to its own docstrings
```python
class Foo(object):
    """
    First line of docstring with class description
    
    Attributes
    ----------
    attr
        sum of first two parameters
    """
    def __init__(self, param1, param2):
        """
        Documentation of __init__
        
        Parameters
        ----------
        param1
            first param of __init__
        param2
            second param of __init__
        """ 
        self.attr = param1 + param2
        return
```
