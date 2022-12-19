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

def delete_minicubes(mask, frac_mis, ncubes, verbose=1):
    """
    delete originally observed data additionally to the missing data for 
    cross-validation or verification of the gapfilling framework.

    Parameters
    ----------
    mask: boolean xarray dataarray with 3 dimensions called time, lat and lon.
          where the corresponding values of the datacube that are observed
          are set True, unobserved values are False, and values that do not 
          need to be gap-filled (e.g. ocean) are set NaN.

    ncubes: int, number of cubes along each axis. Larger ncubes creates smaller
          holes in the data, and vice versa. E.g. if set to 20, the whole 
          dataset is divided into 20 minicubes along each axis (lat, lon, time),
          i.e. 8000 cubes in total

    frac_mis: float, between 0 and 1, fraction of missing values created with 
          the minicubes, as a fraction of all observed values in the dataset.

    Returns
    ----------
    mask: same as input mask, with additional values set to True to achieve the
        desired frac_mis, with minicubes of the size of ncubes.
    """

    if frac_mis < 0 or frac_mis > 1:
        raise AttributeError(f'frac_mis needs to be between 0 and 1 but is {frac_mis}')

    # calculate number of observed and missing values (on land)
    n_mis = mask.sum().item()
    n_land = np.logical_not(np.isnan(mask)).sum().item()
    n_obs = n_land - n_mis
    if n_obs == 0:
        raise ValueError('No observed values on land in data')

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
    cubes_on_land = np.unique(minicubes)
    cubes_on_land = cubes_on_land[~np.isnan(cubes_on_land)]

    # delete randomly X% of the minicubes
    mask_verification = mask.copy(deep=True)
    exitflag = False
    while True:
        selected_cube = np.random.choice(cubes_on_land)
        mask_verification = mask_verification.where(minicubes != selected_cube, True)
        n_cv = mask_verification.sum().load().item() - n_mis
        frac_cv = n_cv / n_obs
        if verbose >= 2:
            print(f'fraction crossval data from observed data: {n_cv} {frac_cv}')
        if frac_cv > frac_mis: 
            if verbose >= 1:
                print(f'fraction crossval data from observed data: {frac_cv}')
            break

    return mask_verification

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
