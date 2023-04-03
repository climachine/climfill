"""
Copyright 2021 ETH Zurich, author: Verena Bessenbacher

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

This file gives you the function for deleting parts of the observed data
for verification or cross-validation of the gap-filling. The function returns
the mask of missing values with added 'minicubes' of originally observed
data that you can set missing now using this mask.
"""

import numpy as np
import xarray as xr

def create_minicubes(mask, ncubes, verbose=1):
    """
    """

    # create minicubes of observed data for cross-validation
    nt = len(mask.time) 
    nx, ny = len(mask.lon), len(mask.lat)
    a = np.arange(ncubes**3).reshape(ncubes,ncubes,ncubes)
    b = a.repeat(np.ceil(nt/ncubes),0).repeat(np.ceil(ny/ncubes),1).repeat(np.ceil(nx/ncubes),2)
    b = b[:nt,:ny,:nx] # trim

    # wrap around xarray
    minicubes = xr.full_like(mask, np.nan) # to xarray for .isin fct
    minicubes[:] = b

    # check only those on land for faster convergence
    minicubes = minicubes.where(~np.isnan(mask)) # only consider cubes on land

    return minicubes
