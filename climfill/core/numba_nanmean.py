"""
this file contains a fast, cython-based version for the spatiotemporal
filtering of the data. 

Adapted after source: 
ilovesymposia.com/2017/03/12/scipys-new-lowlevelcallable-is-a-game-changer/
"""

import numpy as np
from numba import cfunc, carray
from numba.types import intc, intp, float64, voidptr
from numba.types import CPointer

# mean of footprint as I need it
@cfunc(intc(CPointer(float64), intp,
            CPointer(float64), voidptr))
def nbnanmean(values_ptr, len_values, result, data):
    values = carray(values_ptr, (len_values,), dtype=float64)
    result[0] = np.nan
    tmp = 0
    i = 0
    for v in values:
        if ~np.isnan(v):
            tmp = tmp + v
            i = i + 1
    if i != 0:
        result[0] = tmp / max(i,1)
    return 1
